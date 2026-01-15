//! # ddddocr - Rust implementation of OCR for captcha recognition
//!
//! This library is a Rust port of the Python [ddddocr](https://github.com/huashi666/ddddocr) library,
//! designed for offline local captcha recognition. It uses ONNX Runtime for inference and
//! supports recognition of various captcha types including text and character-based captchas.
//!
//! ## Features
//!
//! - Offline local recognition - no network calls required
//! - Support for various captcha types
//! - Based on deep learning models trained on synthetic data
//! - Simple API with minimal dependencies
//! - Async inference support
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use ddddocr::DdddOcr;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Initialize the OCR with the model file
//!     let mut ocr = DdddOcr::new("ddddocr.onnx")?;
//!
//!     // Read the captcha image
//!     let image_bytes = std::fs::read("captcha.png")?;
//!
//!     // Perform recognition
//!     let result = ocr.classification(&image_bytes).await?;
//!
//!     println!("Recognized text: {}", result);
//!     Ok(())
//! }
//! ```
//!
//! ## Error Handling
//!
//! The library uses a custom error type [`DdddOcrError`] for all operations:
//!
//! ```rust,no_run
//! use ddddocr::{DdddOcr, DdddOcrError};
//!
//! async fn recognize_captcha(image_data: &[u8]) -> Result<String, DdddOcrError> {
//!     let mut ocr = DdddOcr::new("ddddocr.onnx")?;
//!     ocr.classification(image_data).await
//! }
//! ```

use {
    image::{imageops::FilterType, load_from_memory, ImageError},
    ort::{
        session::{RunOptions, Session},
        value::Value,
        Error as OrtError,
    },
    std::{collections::HashMap, path::Path},
    thiserror::Error,
};

/// Error type for ddddocr operations.
///
/// This enum wraps various error types that can occur during OCR processing,
/// including image loading errors and ONNX Runtime errors.
#[derive(Debug, Error)]
pub enum DdddOcrError {
    /// Error that occurred during image loading or processing
    #[error("Image error: {0}")]
    Image(#[from] ImageError),
    /// Error that occurred during ONNX Runtime operations
    #[error("ONNX Runtime error: {0}")]
    Ort(#[from] OrtError),
}

/// Character set data for OCR decoding.
///
/// This array maps character indices to their corresponding string representations,
/// used to decode the output tensor from the ONNX model.
const CHARSET_DATA: [&str; 8210] = include!("../charset.json");

/// Main OCR struct for captcha recognition.
///
/// This struct encapsulates the ONNX Runtime session and provides methods for
/// recognizing text in captcha images.
///
/// # Examples
///
/// ```rust,no_run
/// use ddddocr::DdddOcr;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let mut ocr = DdddOcr::new("ddddocr.onnx")?;
///     let image_bytes = std::fs::read("test.png")?;
///     let result = ocr.classification(&image_bytes).await?;
///     println!("{}", result);
///     Ok(())
/// }
/// ```
pub struct DdddOcr {
    /// ONNX Runtime session for running inference
    session: Session,
}

impl DdddOcr {
    /// Creates a new `DdddOcr` instance by loading an ONNX model from the specified path.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the ONNX model file (.onnx)
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the initialized `DdddOcr` instance or an error if loading fails.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The model file cannot be read
    /// - The ONNX Runtime session cannot be created
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use ddddocr::DdddOcr;
    ///
    /// let ocr = DdddOcr::new("ddddocr.onnx")?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new<P>(model_path: P) -> Result<Self, DdddOcrError>
    where
        P: AsRef<Path>,
    {
        // Load ONNX session
        let session = Session::builder()?.commit_from_file(model_path)?;

        Ok(DdddOcr { session })
    }

    /// Performs OCR recognition on the provided image data.
    ///
    /// This method processes the image through the following steps:
    /// 1. Decodes the image from bytes
    /// 2. Resizes the image to a height of 64 pixels while maintaining aspect ratio
    /// 3. Converts to grayscale
    /// 4. Normalizes pixel values: `(pixel / 255.0 - 0.5) / 0.5`
    /// 5. Runs inference through the ONNX model
    /// 6. Decodes the output using CTC (Connectionist Temporal Classification)
    ///
    /// # Arguments
    ///
    /// * `img` - Raw image bytes (e.g., from reading a PNG/JPG file)
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the recognized text as a `String` or an error if processing fails.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The image cannot be decoded
    /// - Image processing fails
    /// - ONNX Runtime inference fails
    /// - Output tensor cannot be extracted or decoded
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use ddddocr::DdddOcr;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let mut ocr = DdddOcr::new("ddddocr.onnx")?;
    ///     let image_bytes = std::fs::read("captcha.png")?;
    ///     let text = ocr.classification(&image_bytes).await?;
    ///     println!("Recognized: {}", text);
    ///     Ok(())
    /// }
    /// ```
    //noinspection SpellCheckingInspection
    pub async fn classification(&mut self, img: &[u8]) -> Result<String, DdddOcrError> {
        // Decode image
        let img = load_from_memory(img)?;

        // Resize to maintain aspect ratio with height = 64
        let new_width = (img.width() as f32 * (64.0 / img.height() as f32)) as u32;
        let resized = img.resize_exact(new_width, 64, FilterType::Lanczos3);

        // Convert to grayscale
        let gray_image = resized.to_luma8();

        // Normalize: convert to float32, /255, then (x-0.5)/0.5
        let height = gray_image.height() as usize;
        let width = gray_image.width() as usize;

        let mut img_data = Vec::with_capacity(height * width);
        for pixel in gray_image.pixels() {
            let normalized = (pixel[0] as f32 / 255.0 - 0.5) / 0.5;
            img_data.push(normalized);
        }

        // Create input tensor: shape [1, 1, height, width] (NCHW format)
        let shape = vec![1usize, 1, height, width];
        let input_value = Value::from_array((shape, img_data))?;

        // Run inference
        let inputs = HashMap::from([("input1".to_string(), input_value)]);
        let run_options = RunOptions::new()?;
        let outputs = self.session.run_async(inputs, &run_options)?.await?;

        // Get output tensor as raw data
        let output = &outputs[0];

        // Try to get raw tensor data - returns (&Shape, &[T])
        let (_, output_data) = output.try_extract_tensor::<i64>()?;

        // Decode results, skipping consecutive duplicates and 0 values
        let mut result = String::new();
        let mut last_item = 0i64;

        for &item in output_data.iter() {
            if item == last_item {
                continue;
            }
            last_item = item;

            if let Some(char_str) = CHARSET_DATA.get(item as usize) {
                result.push_str(char_str);
            }
        }

        Ok(result)
    }
}
