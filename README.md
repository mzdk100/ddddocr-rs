# ddddocr - Rust OCR Library for Captcha Recognition

<div align="center">

**Offline Local Captcha Recognition Library - Rust Edition**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org)

</div>

## Overview

**ddddocr-rs** is a Rust implementation of the popular Python [ddddocr](https://github.com/huashi666/ddddocr) library. It provides offline local captcha recognition using deep learning models trained on synthetic data. The library is designed for testing captcha difficulty and works completely offline without any network calls.

This Rust port maintains the same recognition capabilities while offering Rust's performance and safety benefits.

## Features

- 🔒 **Offline Recognition** - No network calls required, completely local processing
- 🎯 **Multiple Captcha Types** - Supports text-based and character-based captchas
- ⚡ **High Performance** - Built with Rust for speed and efficiency
- 🛡️ **Type Safe** - Full Rust type safety with custom error handling
- 🔧 **Simple API** - Easy to use with minimal setup
- 🚀 **Async Support** - Built-in async inference support

## Installation

Add this to your `Cargo.toml`:

```bash
cargo add ddddocr
```

## Quick Start

### Basic Usage

```rust
use ddddocr::DdddOcr;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the OCR with the model file
    let mut ocr = DdddOcr::new("ddddocr.onnx")?;

    // Read the captcha image
    let image_bytes = std::fs::read("captcha.png")?;

    // Perform recognition
    let result = ocr.classification(&image_bytes).await?;

    println!("Recognized text: {}", result);
    Ok(())
}
```

### Error Handling

The library provides a custom error type for better error handling:

```rust
use ddddocr::{DdddOcr, DdddOcrError};

async fn recognize_captcha(image_data: &[u8]) -> Result<String, DdddOcrError> {
    let mut ocr = DdddOcr::new("ddddocr.onnx")?;
    ocr.classification(image_data).await
}

#[tokio::main]
async fn main() {
    let image_data = std::fs::read("test.png").expect("Failed to read image");
    match recognize_captcha(&image_data).await {
        Ok(text) => println!("Recognized: {}", text),
        Err(e) => eprintln!("Error: {}", e),
    }
}
```

## Model File

You need to have the `ddddocr.onnx` model file. You can obtain it from the original [ddddocr](https://github.com/huashi666/ddddocr) repository or use your own trained model.

## How It Works

The library processes images through the following pipeline:

1. **Image Decoding** - Reads image from bytes (PNG, JPG, etc.)
2. **Resizing** - Resizes to height=64 while maintaining aspect ratio
3. **Grayscale Conversion** - Converts to single channel
4. **Normalization** - Normalizes pixel values: `(pixel/255.0 - 0.5) / 0.5`
5. **Inference** - Runs ONNX model for prediction
6. **CTC Decoding** - Decodes output using CTC (Connectionist Temporal Classification)

## API Reference

### `DdddOcr`

Main struct for OCR operations.

#### Methods

##### `new(model_path: &str) -> Result<Self, DdddOcrError>`

Creates a new `DdddOcr` instance by loading an ONNX model.

- **Parameters:**
  - `model_path`: Path to the ONNX model file (.onnx)
- **Returns:** `Result<DdddOcr, DdddOcrError>`

##### `classification(&mut self, img: &[u8]) -> Result<String, DdddOcrError>`

Performs OCR recognition on image data.

- **Parameters:**
  - `img`: Raw image bytes
- **Returns:** `Result<String, DdddOcrError>` - The recognized text

## Dependencies

- `ort` - ONNX Runtime for Rust
- `image` - Image processing library
- `thiserror` - Error handling

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original [ddddocr](https://github.com/huashi666/ddddocr) Python library by sml2h3
- ONNX Runtime for providing the inference engine
- The Rust community for excellent tooling and libraries

## Note

This library is designed for captcha difficulty testing and educational purposes. The effectiveness of recognition varies depending on captcha types and may require fine-tuning with custom models for specific use cases.

## Links

- [Original Python Library](https://github.com/huashi666/ddddocr)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Rust Language](https://www.rust-lang.org/)
- [中文文档 / 中文文档](README-zh-CN.md)
