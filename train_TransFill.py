import argparse
from collections import namedtuple
import os
import pprint as pp
import random
from typing import Callable, List, NamedTuple, Optional, Tuple, Union
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, ProfilerActivity, schedule
from torch.distributed import init_process_group

from trans_fill2 import TransFill
from sample import randint, sample_example
from tokens import Tokenizer
import operators as op


# Number of times to retry if cuda OOM is encountered.
OOM_RETRIES = 2

# Padding index used for uneven target program lengths.
PADDING_INDEX = 538

# Keys in dict saved to checkpoint file.
# Key for model state.
MODEL_STATE_DICT_KEY = 'model_state_dict'
# Key for optimizer state.
OPT_STATE_DICT_KEY = 'optimizer_state_dict'


# Container for training data.
# (Before and after tokenization)
TokenizedExample = namedtuple(
    'TokenizedExample',
    [
        'examples',
        'strings',
        'programs',
    ],
)


# Configuration for training.
class Config(NamedTuple):
    model: nn.Module
    sample: Callable[[], TokenizedExample]
    optimizer: optim.Optimizer
    clip_grad_value: float
    program_size: int
    device: Optional[Union[torch.device, int]]
    checkpoint_filename: str
    checkpoint_step_size: int
    checkpoint_print_tensors: bool


# Misc info returned by training_step() for logging.
StepInfo = namedtuple(
    'StepInfo',
    [
        'loss',
        'tokenized_example',
        'expected_program_tensor',
        'actual_program_tensor',
    ],
)


def cross_entropy_loss(
        actual: torch.Tensor,
        expected: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-entropy loss ignoring padding tokens due to
    different program lengths.
    """
    program_size = actual.size()[2]
    # Reshape actual_programs (seq length, batch size, program size)
    # to (seq length * batch_size, program size).
    reshaped_actual_programs = actual.view(-1, program_size)
    # Convert expected programs (seq length, batch_size)
    # to a tensor of (seq length * batch_size).
    expected_programs = expected.view(-1)
    loss = F.cross_entropy(
        reshaped_actual_programs,
        expected_programs,
        ignore_index=PADDING_INDEX,
    )
    return loss


def accuracy(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """
    Compute accuracy of the model.

    :param actual: (seq length, batch size, program size)
    :param expected: (seq length, batch size)
    """
    # (seq length, batch size)
    predicted = torch.argmax(actual, dim=2)
    #print(f'predicted{predicted}')
    # (batch size)
    correct = torch.sum(predicted == expected, dim=0)
    #print(f'expected: {expected}')
    # (batch size)
    total = torch.sum(expected != PADDING_INDEX, dim=0)
    return torch.mean(correct / total).item()


class Trainer:
    def __init__(self, config: Config):
        self.config = config

    def load_checkpoint_if_exists(self) -> bool:
        """
        Load state from checkpoint file if it exists.

        returns: True if checkpoint was loaded, false otherwise.
        """
        if (self.config.checkpoint_filename is not None
           and os.path.exists(self.config.checkpoint_filename)):
            print('Starting model from existing checkpoint file: '
                  f'{self.config.checkpoint_filename}')
            loaded = torch.load(self.config.checkpoint_filename)
            self.config.model.load_state_dict(loaded[MODEL_STATE_DICT_KEY])
            self.config.optimizer.load_state_dict(loaded[OPT_STATE_DICT_KEY])
            return True
        return False

    @staticmethod
    def _max_program_length(expected_programs: List[List[int]]) -> int:
        """Return length of longest program."""
        return max([len(program) for program in expected_programs])

    def run_batch(self) -> StepInfo:
        """Execute a single forward-backward pass on a minibatch."""
        tokenized_example = self.config.sample()
        max_length = Trainer._max_program_length(tokenized_example.programs)

        # Convert expected programs from list of lists of ints (uneven lengths)
        # to a tensor of (max sequence_length, batch_size) with padding index.
        l = [
                [
                    program[i] if i < len(program) else PADDING_INDEX
                    for program in tokenized_example.programs
                ]
                for i in range(max_length)
        ]
        L = [len(l[0])*[539]]
        L.extend(l)
        padded_expected_programs = torch.tensor(L, device=self.config.device)
        #eos_tensor = 539 * torch.ones((1, padded_expected_programs.shape[-1]), dtype=torch.int64).to(self.config.device)
        #padded_expected_programs = torch.concat([eos_tensor, padded_expected_programs], dim=0)

        actual_programs = self.config.model(
            tokenized_example.strings,
            target=padded_expected_programs,
            device=self.config.device)


        #print(actual_programs)
        loss = cross_entropy_loss(
            actual=actual_programs,
            expected=padded_expected_programs)

        self.config.optimizer.zero_grad()

        loss.backward()
        nn.utils.clip_grad_value_(
            self.config.model.parameters(),
            clip_value=self.config.clip_grad_value)
        self.config.optimizer.step()

        return StepInfo(
            loss=loss,
            tokenized_example=tokenized_example,
            expected_program_tensor=padded_expected_programs,
            actual_program_tensor=actual_programs)

    def _save_checkpoint(self, step_info: StepInfo, example_idx: int) -> None:
        """Save training state."""
        with torch.no_grad():
            print('Checkpointing at example {}'.format(example_idx))
            print('Loss: {}'.format(step_info.loss))
            print('Accuracy: {}'.format(accuracy(
                step_info.actual_program_tensor,
                step_info.expected_program_tensor)))
            if self.config.checkpoint_print_tensors:
                print_batch_limit = 3

                print('Examples:')
                pp.pprint(
                    step_info.tokenized_example.strings[:print_batch_limit])

                print('Expected programs:')
                print(step_info.tokenized_example.programs[:print_batch_limit])

                print('Actual programs:')
                print(
                    F.softmax(step_info.actual_program_tensor, dim=2)
                    .transpose(1, 0)[:print_batch_limit, :, :]
                )

        if self.config.checkpoint_filename is not None:
            print('Saving to file {}'.format(
                self.config.checkpoint_filename))
            torch.save({
                MODEL_STATE_DICT_KEY: self.config.model.state_dict(),
                OPT_STATE_DICT_KEY: self.config.optimizer.state_dict(),
            }, self.config.checkpoint_filename)

        print('Done')

    def train(self) -> None:
        """Infinite loop for training."""
        example_idx = 0
        while True:
            for i in range(OOM_RETRIES + 1):
                try:
                    step_info = self.run_batch()
                    break
                except torch.cuda.OutOfMemoryError:
                    if i == OOM_RETRIES:
                        raise
                    print('Out of memory, retrying')

            # When training with DDP, only one process should checkpoint.
            checkpointable = (not isinstance(self.config.device, int) or
                              self.config.device == 0)
            if (checkpointable
               and example_idx % self.config.checkpoint_step_size == 0):
                self._save_checkpoint(step_info, example_idx)

            example_idx += 1


def generate_program(batch_size: int) -> List[List[int]]:
    """Generate some simple and short programs for dry-run training."""
    return [
        # Only two programs.
        [0] if randint(0, 2) == 0 else [1, 0]
        for _ in range(batch_size)
    ]


def generate_data(
        program_batch: List[List[int]],
        num_examples: int,
        string_size: int) -> List[List[Tuple[List[int], List[int]]]]:
    """
    Generate some input-output data for our simple and short programs
    for dry-run training.

    Batch is a list (batch_size) of tuples (input, output) of
    list (sequence_length) of token indices.
    """
    batch = []
    for program in program_batch:
        examples = []
        for _ in range(num_examples):
            input_sequence = [randint(0, string_size)]

            # Only two programs here (copy and copy-twice).
            if program == [0]:
                output_sequence = input_sequence
            elif program == [1, 0]:
                output_sequence = input_sequence * 2
            else:
                raise ValueError('Invalid program {}'.format(program))

            examples.append((input_sequence, output_sequence))

        batch.append(examples)

    return batch


def sample_easy(
        batch_size: int,
        string_size: int,
        num_examples: int) -> TokenizedExample:
    """
    Sample simple and short programs and example input-output data for
    dry-run training.
    """
    programs = generate_program(batch_size)
    examples = generate_data(programs, num_examples, string_size)
    return TokenizedExample(
        examples=None,
        programs=programs,
        strings=examples,
    )


def easy_config(seed: Optional[int] = None) -> Config:
    """
    Return config for smaller model on simple and short programs
    as dry-run.
    """
    if seed is not None:
        torch.manual_seed(seed)

    string_size = 3
    program_size = 2
    model = TransFill(
        string_size=string_size,
        string_embedding_size=2,
        hidden_size=8,
        program_size=program_size,
        num_encoder_heads = 4
    )


    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    def sample():
        return sample_easy(
            batch_size=32,
            string_size=string_size,
            num_examples=2,
        )

    return Config(
        model=model,
        optimizer=optimizer,
        clip_grad_value=1.0,
        sample=sample,
        program_size=program_size,
        device=None,  # CPU training.
        checkpoint_filename=None,
        checkpoint_step_size=100,
        checkpoint_print_tensors=True,
    )


def sample_full(
        tokenizer: Tokenizer,
        batch_size: int,
        max_expressions: int,
        max_characters: int) -> TokenizedExample:
    """Sample a batch of programs and example input-output data."""
    example_batch, program_batch, strings_batch = [], [], []

    for _ in range(batch_size):
        example = sample_example(
            max_expressions=max_expressions,
            max_characters=max_characters,
        )
        program = example.program.to_tokens(tokenizer.op_token_table)
        strings = [
            (tokenizer.tokenize_string(input_),
             tokenizer.tokenize_string(output))
            for input_, output in example.strings
        ]
        example_batch.append(example)
        program_batch.append(program)
        strings_batch.append(strings)

    return TokenizedExample(
        examples=example_batch,
        programs=program_batch,
        strings=strings_batch,
    )


def full_config(
        rank: Optional[int] = None,
        seed: Optional[int] = None) -> Config:
    """
    Return config for full model on programs and example input-output data.
    """
    if seed is not None:
        torch.manual_seed(seed)

    tokenizer = Tokenizer.create()

    program_size = len(tokenizer.op_token_table)
    checkpoint_filename = './checkpoint3.pth'
    model = TransFill(
        string_size=len(op.CHARACTER),
        string_embedding_size=128,
        hidden_size=252,
        program_size=program_size,
        num_encoder_heads=4,
        pad_idx_src=75,
        pad_idx_p=538
    )
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    def sample():
        return sample_full(
            tokenizer,
            batch_size=100,
            max_expressions=10,
            max_characters=32,
        )

    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model.to(device)
        print('Using device `cuda`')


    return Config(
        model=model,
        optimizer=optimizer,
        clip_grad_value=1.0,
        program_size=program_size,
        sample=sample,
        device=device,
        checkpoint_filename=checkpoint_filename,
        checkpoint_step_size=10,
        checkpoint_print_tensors=False,
    )

def main() -> None:
    """
    Main function responsible for parsing command line arguments and
    invoking model training.
    """
    config = full_config(seed=1337)
    trainer = Trainer(config)
    loaded = trainer.load_checkpoint_if_exists()
    if loaded:
        # If we are resuming, we don't want to repeat the same
        # training data.
        torch.manual_seed(random.randint(0, 2**32 - 1))
    trainer.train()


if __name__ == '__main__':
    main()
