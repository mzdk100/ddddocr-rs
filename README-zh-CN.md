# ddddocr - Rust OCR 验证码识别库

<div align="center">

**通用验证码离线本地识别 SDK - Rust 版**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org)

</div>

## 简介

**ddddocr-rs** 是 Python 版 [ddddocr](https://github.com/huashi666/ddddocr) 库的 Rust 实现。它使用通过批量随机数据进行深度网络训练的模型，提供离线本地验证码识别功能。该库专为验证码厂商测试自家验证码难易程度而设计，完全离线工作，无需任何网络调用。

这个 Rust 移植版保持了相同的识别能力，同时提供了 Rust 的高性能和安全性优势。

## 特性

- 🔒 **离线识别** - 无需网络调用，完全本地处理
- 🎯 **多种验证码类型** - 支持文本型和字符型验证码
- ⚡ **高性能** - 使用 Rust 构建，速度快、效率高
- 🛡️ **类型安全** - 完整的 Rust 类型安全和自定义错误处理
- 🔧 **简单易用** - API 简洁，设置简单
- 🚀 **异步支持** - 内置异步推理支持

## 安装

在 `Cargo.toml` 中添加：

```bash
cargo add ddddocr
```

## 快速开始

### 基本使用

```rust
use ddddocr::DdddOcr;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 使用模型文件初始化 OCR
    let mut ocr = DdddOcr::new("ddddocr.onnx")?;

    // 读取验证码图片
    let image_bytes = std::fs::read("captcha.png")?;

    // 执行识别
    let result = ocr.classification(&image_bytes).await?;

    println!("识别结果: {}", result);
    Ok(())
}
```

### 错误处理

库提供了自定义错误类型以便更好地处理错误：

```rust
use ddddocr::{DdddOcr, DdddOcrError};

async fn recognize_captcha(image_data: &[u8]) -> Result<String, DdddOcrError> {
    let mut ocr = DdddOcr::new("ddddocr.onnx")?;
    ocr.classification(image_data).await
}

#[tokio::main]
async fn main() {
    let image_data = std::fs::read("test.png").expect("读取图片失败");
    match recognize_captcha(&image_data).await {
        Ok(text) => println!("识别结果: {}", text),
        Err(e) => eprintln!("错误: {}", e),
    }
}
```

## 模型文件

你需要有 `ddddocr.onnx` 模型文件。可以从原始的 [ddddocr](https://github.com/huashi666/ddddocr) 仓库获取，或者使用你自己训练的模型。

## 工作原理

库通过以下管道处理图片：

1. **图片解码** - 从字节数据读取图片（PNG、JPG 等）
2. **调整大小** - 调整到高度=64，同时保持宽高比
3. **灰度转换** - 转换为单通道
4. **归一化** - 归一化像素值：`(像素值/255.0 - 0.5) / 0.5`
5. **推理** - 运行 ONNX 模型进行预测
6. **CTC 解码** - 使用 CTC（连接时序分类）解码输出

## API 参考

### `DdddOcr`

OCR 操作的主结构体。

#### 方法

##### `new(model_path: &str) -> Result<Self, DdddOcrError>`

通过加载 ONNX 模型创建新的 `DdddOcr` 实例。

- **参数：**
  - `model_path`：ONNX 模型文件路径（.onnx）
- **返回值：** `Result<DdddOcr, DdddOcrError>`

##### `classification(&mut self, img: &[u8]) -> Result<String, DdddOcrError>`

对图片数据执行 OCR 识别。

- **参数：**
  - `img`：原始图片字节数据
- **返回值：** `Result<String, DdddOcrError>` - 识别的文本

## 依赖项

- `ort` - Rust 版 ONNX Runtime
- `image` - 图片处理库
- `thiserror` - 错误处理

## 开源协议

本项目采用 MIT 协议 - 详见 [LICENSE](LICENSE) 文件。

## 致谢

- [ddddocr](https://github.com/huashi666/ddddocr) 原始 Python 库，作者 sml2h3
- ONNX Runtime 提供推理引擎
- Rust 社区提供的优秀工具和库

## 说明

本库专为验证码难度测试和教育目的而设计。识别效果因验证码类型而异，对于特定用例可能需要使用自定义模型进行微调。

## 原始项目

本项目是 [ddddocr](https://github.com/huashi666/ddddocr) Python 库的 Rust 移植版本。原始 Python 库由 sml2h3 开发，奉行开箱即用、最简依赖的理念，尽量减少用户的配置和使用成本。

### 原始项目特点

- 开箱即用，无需复杂配置
- 最简依赖设计
- 适合验证码厂商测试自家验证码强度
- 使用大量随机合成数据训练而成

## 相关链接

- [原始 Python 库](https://github.com/huashi666/ddddocr)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Rust 语言](https://www.rust-lang.org/)
- [English Documentation / 英文文档](README.md)
