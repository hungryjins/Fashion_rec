# Fashion Recommendation System

An AI-powered fashion recommendation system that combines text and image search capabilities using advanced machine learning models.

## 🚀 Features

- **Multi-modal Search**: Text-based and image-based fashion item search
- **Hybrid Search**: Combine text and image inputs for enhanced recommendations
- **AI Image Generation**: Create new fashion items using DALL-E 3
- **Real-time Processing**: Fast search and recommendation using vector databases
- **Web Interface**: Streamlit-based user-friendly web application

## 🏗️ Architecture

### Core Technologies

- **Vector Database**: Pinecone for efficient similarity search
- **Computer Vision**:
  - CLIP (fashion-tuned) for image-text embeddings
  - YOLO for fashion item detection
  - SPLADE for sparse vector search
- **Language Models**: GPT-4 for text processing and image generation
- **Web Framework**: Streamlit for the user interface

### System Components

```
📁 Fashion_rec/
├── 📁 app/                    # Main application
│   ├── demo_app.py           # Streamlit web app
│   ├── search_method_wrapper.py  # Search method wrapper
│   ├── search_utils.py       # Search utilities
│   ├── image_utils.py        # Image processing
│   ├── yolo_utils.py         # YOLO object detection
│   └── setup_req.py          # Model initialization
├── 📊 Jupyter Notebooks      # Development and analysis
└── 📁 test_images/           # Test images
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)

### Setup

1. Clone the repository

```bash
git clone <repository-url>
cd Fashion_rec
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Set up API keys

```bash
export OPENAI_API_KEY="your_openai_api_key"
export PINECONE_API_KEY="your_pinecone_api_key"
```

## 📖 Usage

### Web Application

Run the Streamlit app:

```bash
cd app
streamlit run demo_app.py
```

### Jupyter Notebooks

The project includes several Jupyter notebooks for different functionalities:

1. **Data Preparation** (`00. fashionpedia_Data_prep.ipynb`)

   - Fashionpedia dataset preprocessing
   - Image cropping and annotation processing

2. **Text Search** (`03. Text_input_module.ipynb`)

   - Text-based fashion item search
   - Query transformation and filtering

3. **Image Search** (`04. Image_input_module.ipynb`)

   - Image-based fashion item search
   - Object detection and feature extraction

4. **Hybrid Search** (`05. Hybrid_search_module.ipynb`)
   - Combined text and image search
   - Multi-modal recommendation

## 🔧 Configuration

### API Keys

Before running the application, you need to set up the following API keys:

1. **OpenAI API Key**: For GPT-4 and DALL-E 3
2. **Pinecone API Key**: For vector database access

Replace the placeholder values in the code:

- `YOUR_OPENAI_API_KEY` → Your actual OpenAI API key
- `YOUR_PINECONE_API_KEY` → Your actual Pinecone API key

### Model Setup

The system uses several pre-trained models:

- Fashion-tuned CLIP: `patrickjohncyh/fashion-clip`
- YOLO for fashion detection: `valentinafeve/yolos-fashionpedia`
- SPLADE for sparse search: `naver/splade-cocondenser-ensembledistil`

## 📊 Data

The system is designed to work with the Fashionpedia dataset, which includes:

- Fashion item images with annotations
- Category and attribute labels
- Bounding box coordinates for object detection

## 🔍 Search Methods

### 1. Text Search

- Natural language fashion queries
- Category-based filtering
- Attribute-based refinement

### 2. Image Search

- Upload fashion item images
- Automatic object detection
- Similar item recommendation

### 3. Hybrid Search

- Combine text and image inputs
- Multi-stage processing pipeline
- Enhanced recommendation accuracy

## 🤖 AI Features

### Image Generation

- DALL-E 3 integration for fashion item creation
- Style transformation based on user input
- Real-time image generation

### Query Processing

- GPT-4 powered query understanding
- Fashion-specific query transformation
- Multi-gateway validation system

## 📈 Performance

- **Search Speed**: Sub-second response times
- **Accuracy**: High precision fashion item matching
- **Scalability**: Vector database for efficient large-scale search

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Fashionpedia dataset providers
- OpenAI for GPT-4 and DALL-E 3
- Pinecone for vector database services
- Hugging Face for pre-trained models

## 📞 Support

For questions and support, please open an issue in the GitHub repository.
