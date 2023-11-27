from typing import Any, List, Optional, Tuple, Union
from torch.nn.utils.rnn import (
    PackedSequence,
    pad_sequence,
    pack_padded_sequence,
    pad_packed_sequence,
)


import torch
from torch import nn, optim
import torch.nn.functional as F
import math
import numpy as np


def positional_encoding(seq_len, d_model, device):
  pe = np.zeros((seq_len, d_model))
  for pos in range(seq_len):
    for i in range(0, d_model, 2):
      pe[pos, i] = math.sin(pos / (10000 ** (i/d_model)))
      pe[pos, i+1] = math.cos(pos / (10000 ** (i/d_model)))
  return torch.from_numpy(pe[np.newaxis, :]).to(device)


class TransFill(nn.Module):
    def __init__(
            self,
            string_size: int,
            string_embedding_size: int,
            hidden_size: int,
            program_size: int,
            num_encoder_heads: int,
            pad_idx_src=75,
            pad_idx_p=-1
    ):
        """
        Implements the RobustFill program synthesis model.
        :param string_size: The number of tokens in the string vocabulary. #len(op.CHARACTER) 75
        :param string_embedding_size: The size of the string embedding.    #128
        :param hidden_size: The size of the hidden states of the           #512
            input/output encoders and decoder.
        :param program_size: The number of tokens in the program output.   #538
        :param num_encoder_heads:
        :param pad_idx_src:
        :param pad_idx_p:
        """
        super().__init__()
        self.pad_idx_src = pad_idx_src
        self.pad_idx_p = pad_idx_p

        self.embedding = nn.Embedding(string_size+1, string_embedding_size) #  75 -> 128
        self.linear = nn.Linear(string_embedding_size, hidden_size) # 128 -> 512
        self.dropout = nn.Dropout(0.4)

#################################################added
        self.softmax = nn.Softmax(dim=0)

        self.I_transformer_encoder = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_encoder_heads,
            dim_feedforward=2048,
            dropout=0.4,
            batch_first=False
        )
        #self.I_energy = nn.Linear(hidden_size,1)

        self.O_transformer_encoder = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_encoder_heads,
            dim_feedforward=2048,
            dropout=0.4,
            batch_first=False
        )
        #self.O_energy = nn.Linear(hidden_size,1)

        self.IO_transformer_encoder = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_encoder_heads,
            dim_feedforward=2048,
            dropout=0.4,
            batch_first=False
        )
        #self.IO_energy = nn.Linear(hidden_size,1)

        self.transformer_decoder = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_encoder_heads,
            dim_feedforward=2048,
            dropout=0.4,
            batch_first=False
        )

#######################################################################################

        self.program_decoder = ProgramDecoder(
            hidden_size = hidden_size,
            program_size = program_size,
            tgt_padding_index=self.pad_idx_p,
            num_decoder_heads=num_encoder_heads
        ) #512 -> 538

    @staticmethod
    def _check_num_examples(batch) -> int:
        """Check that the numbers of examples are consistent across batches."""
        assert len(batch) > 0
        num_examples = len(batch[0])
        assert all([
            len(examples) == num_examples
            for examples in batch
        ])
        return num_examples

    @staticmethod
    def _split_flatten_examples(batch: List) -> Tuple[List, List]:
        """
        Flatten the examples so that they just separate data in the same batch.
        They will be integrated again at the max-pool operator.

        :param batch: List (batch_size) of tuples (input, output) of
            lists (sequence_length) of token indices.
        :returns: Tuple of two lists (batch_size * num_examples) of lists
            (sequence_length) of token indices.
        """
        input_batch = [
            input_sequence
            for examples in batch
            for input_sequence, _ in examples
        ]
        output_batch = [
            output_sequence
            for examples in batch
            for _, output_sequence in examples
        ]
        return input_batch, output_batch

    def _embed(
            self,
            batch: List,
            device: Optional[Union[torch.device, int]],
            ) -> Tuple[PackedSequence, torch.Tensor]:
        """
        Convert each list of tokens in a batch into a tensor of
        shape (sequence_length, string_embedding_size).
        """
        lengths = torch.as_tensor([len(v) for v in batch], dtype=torch.int64)
        # (sequence_length, batch_size).
        padded = pad_sequence([
            torch.as_tensor(v)
            for v in batch
        ], batch_first=False, padding_value=self.pad_idx_src) #, batch_first=False, padding_value=self.pad_idx_src

        padded = padded.to(device)

        # (sequence_length, batch_size, string_embedding_size).
        embedded = self.embedding(padded)
        # (sequence_length, batch_size, hidden_dim).
        embedded = self.dropout(self.linear(embedded))


        ####################### where u wanna change
        #packed = pack_padded_sequence(embedded, lengths, enforce_sorted=False)
        # This has to be after pack_padding_sequence because it expects
        # length to be on the CPU.
        lengths = None
        ###############################


        return embedded, padded, lengths  # embedded: (sequence_length, batch_size*4, string_embedding_size).



    def _make_padding_mask(self, padded, pad_idx, device):
        """
        Create the src_key_padding_mask for a batch of padded input sequences.

        Args:
            padded_input (torch.Tensor): Padded input sequences of shape (sequence_length, batch_size).
            padding_index (int): The padding index value used in the input sequences.

        Returns:
            torch.Tensor: src_key_padding_mask of shape (batch_size, sequence_length).
        """
        # Check if the input is a PyTorch tensor
        if not isinstance(padded, torch.Tensor):
            raise ValueError("Input must be a PyTorch tensor.")

        # Check that the dimensions match the expected format (sequence_length, batch_size)
        if len(padded.shape) != 2:
            raise ValueError("Input must have shape (sequence_length, batch_size).")

        # Create a mask by comparing each element with the padding_index
        src_key_padding_mask = (padded == pad_idx).bool()

        # Transpose the mask to have shape (batch_size, sequence_length)
        src_key_padding_mask = src_key_padding_mask.transpose(0, 1)

        return src_key_padding_mask.to(device)



    def _make_attention_pooling(self, incoded_v, energy_linear, mask):
        # incoded_v: (sequence_length, batch_size, hidden_dim)

        energies = energy_linear(incoded_v) # (sequence_length, batch_size, 1)
        #print(mask.transpose(0,1).unsqueeze(-1).shape)
        #print(energies.shape)
        energies_maksed = energies.masked_fill(mask.transpose(0,1).unsqueeze(-1), float("-1e20"))
        attentions = self.softmax(energies_maksed) # (sequence_length, batch_size, 1)


        # attentions : (sequence_length, batch_size, 1)  sbv
        # incoded_v.shape : (sequence_length, batch_size, hidden_dim)  sbh
        # hidden.shape : (1, batch_size, hidden_dim) vbh
        hidden = torch.einsum("sbv,sbh->vbh" ,attentions, incoded_v)

        return hidden.squeeze(0)  #(batch_size, hidden_dim)



    def encode(
            self,
            batch: List,
            device: Optional[Union[torch.device, int]] = None,
            ) -> Tuple[
                Tuple[torch.Tensor, torch.Tensor],
                Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode input-output pairs.

        :param batch: List (batch_size) of tuples (input, output) of
            list (sequence_length) of token indices.
        :param device: The device to send the input data to.
        :returns: Tuple of (final hidden state, all hidden states) of encoding.
        """
        #print(device)
        input_batch, output_batch = TransFill._split_flatten_examples(batch)

        # Encode inputs.
        embedded_input, padded_input, input_seq_lengths = self._embed(
            input_batch,
            device=device)

        #print(padded_input)
        I_mask_src = self._make_padding_mask(padded=padded_input, pad_idx=self.pad_idx_src, device=device)
        #print(I_mask_src)
        input_all_hidden = self.I_transformer_encoder(src=embedded_input, src_key_padding_mask=I_mask_src)
        #input_all_hidden = self.I_transformer_encoder(src=embedded_input)
        # (sequence_length, batch_size, hidden_dim)


        # Encode outputs.
        embedded_output, padded_output, output_seq_lengths = self._embed(
            output_batch,
            device=device)
        O_mask_src = self._make_padding_mask(padded=padded_output, pad_idx=self.pad_idx_src, device=device)
        output_all_hidden = self.O_transformer_encoder(src=embedded_output, src_key_padding_mask=O_mask_src)
        #output_all_hidden = self.O_transformer_encoder(src=embedded_output)
        # (sequence_length, batch_size, hidden_dim)

        IO_mask_src = torch.concat([I_mask_src, O_mask_src], dim=-1)
        stacked_encoded_IO = torch.concat([input_all_hidden, output_all_hidden], dim=0)
        IO_all_hidden = self.IO_transformer_encoder(src=stacked_encoded_IO, src_key_padding_mask=IO_mask_src)
        #IO_all_hidden = self.IO_transformer_encoder(src=stacked_encoded_IO)
        # (sequence_length_I + sequence_length_O, batch_size, hidden_dim)





        return IO_all_hidden, IO_mask_src  # (batch_size, hidden_dim), (sequence_length_I + sequence_length_O, batch_size, hidden_dim)

    def forward(
            self,
            batch: List,
            target: torch.Tensor,
            device: Optional[Union[torch.device, int]] = None) -> torch.Tensor:
        """
        Forward pass through RobustFill.

        :param batch: List (batch_size) of tuples (input, output) of
            list (sequence_length) of token indices.
        :param target: The target program output used for
            teacher-forcing during decoding.
            (sequence_length, batch_size)
        :param device: The device to send the input data to.
        """
        num_examples = TransFill._check_num_examples(batch)
        all_hidden, IO_mask_src = self.encode(
            batch=batch,
            device=device,
        )


        return self.program_decoder(
            memory=all_hidden,
            memory_mask = IO_mask_src,
            num_examples=num_examples,
            target=target,
            device=device,
        )



class ProgramDecoder(nn.Module):
    """Program decoder module."""

    def __init__(self, hidden_size, program_size, tgt_padding_index, num_decoder_heads):
        super().__init__()

        self.program_size = program_size+2
        self.hidden_size = hidden_size
        self.tgt_padding_index = tgt_padding_index

        self.tgt_embedding = nn.Embedding(self.program_size, hidden_size)
        self.transformerdecoder = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_decoder_heads,
            dim_feedforward=2048,
            dropout=0.4,
            batch_first=False
        )
        self.dropout = nn.Dropout(0.4)
        self.linear = nn.Linear(hidden_size, self.program_size)
        self.softmax = nn.Softmax(dim=-1)

        self.max_pool_linear = nn.Linear(hidden_size, hidden_size)
        self.linea_classifier = nn.Linear(hidden_size, self.program_size)

    def _make_padding_mask(self, padded, pad_idx, device):
        """
        Create the src_key_padding_mask for a batch of padded input sequences.

        Args:
            padded_input (torch.Tensor): Padded input sequences of shape (sequence_length, batch_size).
            padding_index (int): The padding index value used in the input sequences.

        Returns:
            torch.Tensor: src_key_padding_mask of shape (batch_size, sequence_length).
        """
        # Check if the input is a PyTorch tensor
        if not isinstance(padded, torch.Tensor):
            raise ValueError("Input must be a PyTorch tensor.")

        # Check that the dimensions match the expected format (sequence_length, batch_size)

        if len(padded.shape) != 2:
            raise ValueError("Input must have shape (sequence_length, batch_size).")

        # Create a mask by comparing each element with the padding_index
        tgt_key_padding_mask = (padded == pad_idx).bool()

        # Transpose the mask to have shape (batch_size, sequence_length)
        tgt_key_padding_mask = tgt_key_padding_mask.transpose(0, 1)

        return tgt_key_padding_mask.to(device)


    def _generate_tgt_mask(self, output, device):
        """
        Generate the target mask (tgt_mask) to prevent future tokens from being attended to.

        Args:
            output (torch.Tensor): Output sequence tensor (tgt_seq_len, batch_size, d_model).
            padding_idx (int): Padding index used for padding in output sequences.

        Returns:
            torch.Tensor: Target mask tensor (tgt_seq_len, tgt_seq_len).
        """
        tgt_seq_len = output.size(0)
        tgt_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len), diagonal=1)
        tgt_mask = tgt_mask.masked_fill(tgt_mask == 1, float('-inf'))
        return tgt_mask.to(device)



    def forward(
            self,
            memory:torch.Tensor,
            memory_mask:torch.Tensor,
            num_examples: int,
            target: torch.Tensor,
            device: Optional[Union[torch.device, int]]) -> torch.Tensor:
        """
        Forward pass through the decoder.

        :param hidden: Hidden states of LSTM from output encoder.
        :param output_all_hidden: Entire sequence of hidden states of
            LSTM from output encoder (to be attended to).
        :param num_examples: The number of examples in the batch.
        :param max_program_length: The maximum length of the program
            to generate.
        :param target: The target program output used for
            teacher-forcing during decoding.
            (sequence_length, batch_size, program_size)
        """
        target = target.to(torch.int64)
        tgt_embeddings = self.tgt_embedding(target.to(device)) #(seq_len, batch_size, hidden_size)
        decoder_input = tgt_embeddings.repeat_interleave(num_examples, dim=1)
        tgt_mask = self._generate_tgt_mask(target, device)
        tgt_key_padding_mask = self._make_padding_mask(target, self.tgt_padding_index, device)
        tgt_key_padding_mask_expanded = tgt_key_padding_mask.repeat_interleave(num_examples, dim=0)
        decoder_output = self.transformerdecoder(tgt=decoder_input, memory=memory, tgt_mask=tgt_mask,
                                                 memory_mask=None, tgt_key_padding_mask=tgt_key_padding_mask_expanded,
                                                 memory_key_padding_mask=memory_mask)
        #(seq_len, batch_size*num_examples, hidden_size)

        decoder_output = torch.tanh(self.max_pool_linear(decoder_output))
        decoder_output= decoder_output.view(decoder_output.shape[0], -1, num_examples, self.hidden_size)
        decoder_output= decoder_output.permute(0, 1,  3, 2).contiguous()
        # (seq_len, batch_size, hidden_size, num_examples)
        decoder_output_rolled = decoder_output.reshape(-1, decoder_output.shape[2], decoder_output.shape[3])
        pooled = F.max_pool1d(decoder_output_rolled, num_examples).squeeze(-1)
        # (seq_len*batch_size, hidden_size)
        pooled_unrolled = pooled.reshape(decoder_output.shape[0], decoder_output.shape[1], -1)
        # (seq_len, batch_size, hidden_size)
        probs = self.softmax(self.dropout(self.linea_classifier(pooled_unrolled)))
        # (seq_len, batch_size, program_size)

        return probs


