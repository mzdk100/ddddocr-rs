use {ddddocr::DdddOcr, tokio::fs::read};

#[tokio::test]
async fn test_ocr_classification() {
    // Load the ONNX model and charset
    let mut ocr = DdddOcr::new("ddddocr.onnx").expect("Failed to load OCR model");

    // Load test image
    let image_data = read("test.png").await.expect("Failed to read test image");

    // Perform classification and get string result
    let result = ocr
        .classification(&image_data)
        .await
        .expect("Classification failed");

    assert_eq!("消息", result);
}
