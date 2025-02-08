import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
from tensorflow.keras import layers
from tensorflow.keras.applications import efficientnet
import keras

# Constants
IMAGE_SIZE = (299, 299)
SEQ_LENGTH = 25
EMBED_DIM = 512
FF_DIM = 1024
NUM_HEADS = 6

# ✅ Function to load tokenizer model
@st.cache_resource
def load_tokenizer():
    model_path = "IMC/image_captioning_model"

    if not os.path.exists(model_path):
        st.error(f"Tokenizer model not found at {model_path}. Ensure the model exists.")
        return None

    try:
        # Attempt loading as TFSMLayer (for SavedModel format)
        tokenizer = keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
        return tokenizer
    except Exception as e:
        st.warning(f"TFSMLayer failed: {str(e)}. Trying alternate method...")

    try:
        # Attempt loading as a Keras model (.keras or .h5)
        if os.path.exists(model_path + ".keras"):
            tokenizer = tf.keras.models.load_model(model_path + ".keras", compile=False)
        elif os.path.exists(model_path + ".h5"):
            tokenizer = tf.keras.models.load_model(model_path + ".h5", compile=False)
        else:
            st.error("No valid tokenizer model file found (.keras or .h5).")
            return None
        return tokenizer
    except Exception as e:
        st.error(f"Failed to load tokenizer: {str(e)}")
        return None

# ✅ Function to load CNN model
def get_cnn_model():
    base_model = efficientnet.EfficientNetB0(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet")
    base_model.trainable = False
    base_model_out = layers.Reshape((-1, 1280))(base_model.output)
    cnn_model = tf.keras.models.Model(base_model.input, base_model_out)
    return cnn_model

# ✅ Transformer Encoder Block
class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = layers.Dense(embed_dim, activation="relu")
        self.layernorm_1 = layers.LayerNormalization()

    def call(self, inputs, training, mask=None):
        inputs = self.dense_proj(inputs)
        attention_output = self.attention(query=inputs, value=inputs, key=inputs)
        return self.layernorm_1(inputs + attention_output)

# ✅ Transformer Decoder Block
class TransformerDecoderBlock(layers.Layer):
    def __init__(self, embed_dim, ff_dim, num_heads, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.embedding = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.attention_1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.attention_2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = tf.keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.dropout_1 = layers.Dropout(0.1)
        self.dropout_2 = layers.Dropout(0.5)
        self.out = layers.Dense(vocab_size)

    def call(self, inputs, encoder_outputs, training, mask=None):
        inputs = self.embedding(inputs)
        attention_output_1 = self.attention_1(query=inputs, value=inputs, key=inputs)
        out_1 = self.layernorm_1(inputs + attention_output_1)
        attention_output_2 = self.attention_2(query=out_1, value=encoder_outputs, key=encoder_outputs)
        out_2 = self.layernorm_2(out_1 + attention_output_2)
        proj_output = self.dense_proj(out_2)
        proj_out = self.layernorm_3(out_2 + proj_output)
        return self.out(self.dropout_2(proj_out, training=training))

# ✅ Image Captioning Model
class ImageCaptioningModel(tf.keras.Model):
    def __init__(self, cnn_model, encoder, decoder):
        super().__init__()
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        x = self.cnn_model(inputs[0])
        x = self.encoder(x, False)
        return self.decoder(inputs[2], x, training=inputs[1])

# ✅ Load the image captioning model
@st.cache_resource
def load_model():
    config_path = "IMC/config_train.json"
    weights_path = "IMC/weights.h5"

    if not os.path.exists(config_path) or not os.path.exists(weights_path):
        st.error(f"Missing configuration or weights file in 'IMC/'. Check your files.")
        return None

    with open(config_path) as json_file:
        model_config = json.load(json_file)

    cnn_model = get_cnn_model()
    encoder = TransformerEncoderBlock(embed_dim=model_config["EMBED_DIM"], dense_dim=model_config["FF_DIM"], num_heads=model_config["NUM_HEADS"])
    decoder = TransformerDecoderBlock(embed_dim=model_config["EMBED_DIM"], ff_dim=model_config["FF_DIM"], num_heads=model_config["NUM_HEADS"], vocab_size=model_config["VOCAB_SIZE"])

    caption_model = ImageCaptioningModel(cnn_model, encoder, decoder)
    caption_model.load_weights(weights_path)
    return caption_model

# ✅ Read and preprocess image
def read_image_inf(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.expand_dims(img, axis=0)

# ✅ Generate caption
def generate_caption(image_path, caption_model, tokenizer):
    vocab = tokenizer.get_vocabulary()
    index_lookup = {i: word for i, word in enumerate(vocab)}
    max_length = SEQ_LENGTH - 1

    img = read_image_inf(image_path)
    img_features = caption_model.cnn_model(img)
    encoded_img = caption_model.encoder(img_features, training=False)

    caption = "sos"
    for _ in range(max_length):
        tokenized_caption = tokenizer([caption])[:, :-1]
        predictions = caption_model.decoder(tokenized_caption, encoded_img, training=False)
        predicted_index = np.argmax(predictions[0, -1, :])
        predicted_word = index_lookup[predicted_index]
        if predicted_word == "eos":
            break
        caption += " " + predicted_word

    return caption.replace("sos ", "")

# ✅ Streamlit App
def main():
    st.title("🖼️ Image Captioning with AI")
    st.write("Upload an image and let the model generate a caption!")

    uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        st.write("Generating caption...")

        # Save uploaded image
        temp_path = "temp_image.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        tokenizer = load_tokenizer()
        caption_model = load_model()

        if tokenizer and caption_model:
            caption = generate_caption(temp_path, caption_model, tokenizer)
            st.success(f"**Generated Caption:** {caption}")
        else:
            st.error("Model or tokenizer failed to load.")

if __name__ == "__main__":
    main()
