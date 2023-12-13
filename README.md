# TransFill: Reimplementation of RobustFill with transformers.

RobustFill Original Paper: https://arxiv.org/pdf/1703.07469.pdf




## TransFill Model Architecture

The TransFill model is designed with several key components, each playing a crucial role in the processing and transformation of data. Below is a detailed overview of the architecture, as illustrated in Figure 1.

### Dual Transformer Encoders
- **Function**: Process input and output sequences separately.
- **Description**: Involves two separate Transformer encoder layers, one for the input sequence and the other for the output sequence. This dual-encoder structure is pivotal in capturing and understanding the nuances of both the input and output examples.

### Information Integration Block
- **Function**: Combine and process encoded sequences.
- **Description**: The outputs of the two encoders (input and output sequences) are concatenated and further processed in an additional Transformer encoder layer. This layer is responsible for integrating and encoding the combined information from both sequences, resulting in a comprehensive, cumulated encoding.

### Transformer Decoder
- **Function**: Generate the next sequence in the output.
- **Description**: The integrated encoding is fed into a Transformer decoder, which also receives the current output sequence. This decoder processes the information to generate the subsequent sequence in the output, building the synthesized program step by step.

### Downstream Task for Prediction
- **Function**: Predict subsequent sequence of the program.
- **Description**: The output from the decoder is passed through a downstream task component, which is tasked with predicting the next sequence of the program based on the decoded output. It includes a max-pooling layer to reduce dimensionality, accommodating four I/O inputs to predict one program.

The operational flow of TransFill begins with the dual encoding of input and output sequences, followed by their integration and cumulative encoding. The decoder then takes this integrated encoding, along with the current output sequence, to predict the next part of the program. This iterative process continues until the complete program is synthesized from the provided I/O examples, demonstrating the model’s capability to understand and translate example-based specifications into executable programs.


## Figure 1
![TransFill_model](https://github.com/mohsen-nyb/Trans-Fill/assets/122830808/cf02dc3d-c15e-4e2c-a96c-2a4eb631d0a7)



