# Model Architecture Description

Our strong lens finder relies on an adapted **ResNet18 (Residual Network)** architecture. The standard ResNet18 is composed of 18 deep layers—primarily stacked Convolutional Operations intermeshed with Batch Normalization, ReLU activations, and core Residual connections that combat vanishing gradients.

## Customization for Lens Classification
Because our input consists of multi-filter arrays of shape `(3, 64, 64)` and we aim to perform binary classification:
1. **Input Map**: The default initial Conv2d layer natively supports 3 input channels, reading the raw lens geometries perfectly.
2. **Intermediate Maps**: Through a sequence of sequential pooling and convolution blocks (`layer1` through `layer4`), spatial dimensions condense while extracting deep feature abstractions (ranging from basic edges to complex galactic shapes).
3. **Classification Head**: The canonical ResNet18 output outputs 1000 features for ImageNet. We swapped the final Fully Connected mapping (`nn.Linear`) to channel down into a singular output node `(1 logit)`. Passing this value through a Sigmoid activation models exactly the likelihood that a strong lens is present.

### Architectural Diagram
```mermaid
graph TD
    A[Input Images: 3x64x64] --> B[Conv1 7x7, 64 filters, stride 2]
    B --> C[BatchNorm2d + ReLU]
    C --> D[MaxPool2d 3x3]
    D --> E[Layer 1: 2x Residual Blocks 64 filters]
    E --> F[Layer 2: 2x Residual Blocks 128 filters]
    F --> G[Layer 3: 2x Residual Blocks 256 filters]
    G --> H[Layer 4: 2x Residual Blocks 512 filters]
    H --> I[AdaptiveAvgPool2d 1x1]
    I --> J[Flatten]
    J --> K[Fully Connected Layer: 512 -> 1]
    K --> L[Output Logit]
    L --> M((Sigmoid for Probabilities))
```

*Note: For a visual representation, please refer to the attached `neural_network_visual.png` generated separately.*
