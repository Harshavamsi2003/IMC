import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from tensorflow.keras import layers
from tensorflow.keras.applications import efficientnet

# Constants
IMAGE_SIZE = (299, 299)
SEQ_LENGTH = 25
EMBED_DIM = 512
FF_DIM = 1024
NUM_HEADS = 6

# Load the tokenizer
@st.cache_resource  # Use st.cache_resource for caching objects like models and tokenizers
def load_tokenizer()
    # Load the tokenizer using TFSMLayer if it's in SavedModel format
    tokenizer = tf.keras.layers.TFSMLayer(IMCimage_captioning_model, call_endpoint='serving_default')
    return tokenizer

# Define the CNN model
def get_cnn_model()
    base_model = efficientnet.EfficientNetB0(
        input_shape=(IMAGE_SIZE, 3), include_top=False, weights=imagenet,
    )
    base_model.trainable = False
    base_model_out = layers.Reshape((-1, 1280))(base_model.output)
    cnn_model = tf.keras.models.Model(base_model.input, base_model_out)
    return cnn_model

# Transformer Encoder Block
class TransformerEncoderBlock(layers.Layer)
    def __init__(self, embed_dim, dense_dim, num_heads, kwargs)
        super().__init__(kwargs)
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = layers.Dense(embed_dim, activation=relu)
        self.layernorm_1 = layers.LayerNormalization()

    def call(self, inputs, training, mask=None)
        inputs = self.dense_proj(inputs)
        attention_output = self.attention(query=inputs, value=inputs, key=inputs)
        return self.layernorm_1(inputs + attention_output)

# Transformer Decoder Block
class TransformerDecoderBlock(layers.Layer)
    def __init__(self, embed_dim, ff_dim, num_heads, vocab_size, kwargs)
        super().__init__(kwargs)
        self.embed_dim = embed_dim
        self.attention_1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.attention_2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = tf.keras.Sequential([layers.Dense(ff_dim, activation=relu), layers.Dense(embed_dim)])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.embedding = PositionalEmbedding(embed_dim=EMBED_DIM, sequence_length=SEQ_LENGTH, vocab_size=vocab_size)
        self.out = layers.Dense(vocab_size)
        self.dropout_1 = layers.Dropout(0.1)
        self.dropout_2 = layers.Dropout(0.5)

    def call(self, inputs, encoder_outputs, training, mask=None)
        inputs = self.embedding(inputs)
        inputs = self.dropout_1(inputs, training=training)
        attention_output_1 = self.attention_1(query=inputs, value=inputs, key=inputs)
        out_1 = self.layernorm_1(inputs + attention_output_1)
        attention_output_2 = self.attention_2(query=out_1, value=encoder_outputs, key=encoder_outputs)
        out_2 = self.layernorm_2(out_1 + attention_output_2)
        proj_output = self.dense_proj(out_2)
        proj_out = self.layernorm_3(out_2 + proj_output)
        proj_out = self.dropout_2(proj_out, training=training)
        return self.out(proj_out)

# Positional Embedding
class PositionalEmbedding(layers.Layer)
    def __init__(self, sequence_length, vocab_size, embed_dim, kwargs)
        super().__init__(kwargs)
        self.token_embeddings = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=embed_dim)

    def call(self, inputs)
        positions = tf.range(start=0, limit=tf.shape(inputs)[-1], delta=1)
        return self.token_embeddings(inputs) + self.position_embeddings(positions)

# Image Captioning Model
class ImageCaptioningModel(tf.keras.Model)
    def __init__(self, cnn_model, encoder, decoder)
        super().__init__()
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs)
        x = self.cnn_model(inputs[0])
        x = self.encoder(x, False)
        x = self.decoder(inputs[2], x, training=inputs[1], mask=None)
        return x

# Load the model
@st.cache_resource  # Use st.cache_resource for caching objects like models
def load_model()
    with open(IMCconfig_train.json) as json_file
        model_config = json.load(json_file)

    EMBED_DIM = model_config[EMBED_DIM]
    FF_DIM = model_config[FF_DIM]
    NUM_HEADS = model_config[NUM_HEADS]
    VOCAB_SIZE = model_config[VOCAB_SIZE]

    cnn_model = get_cnn_model()
    encoder = TransformerEncoderBlock(embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=NUM_HEADS)
    decoder = TransformerDecoderBlock(embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=NUM_HEADS, vocab_size=VOCAB_SIZE)
    
    caption_model = ImageCaptioningModel(cnn_model=cnn_model, encoder=encoder, decoder=decoder)
    # caption_model.load_weights(IMCweights.h5)
    caption_model = tf.keras.models.load_model(IMCweights.h5)
    return caption_model

# Read and preprocess the image
def read_image_inf(img_path)
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.expand_dims(img, axis=0)

# Generate caption
def generate_caption(image_path, caption_model, tokenizer)
    vocab = tokenizer.get_vocabulary()
    index_lookup = dict(zip(range(len(vocab)), vocab))
    max_decoded_sentence_length = SEQ_LENGTH - 1

    img = read_image_inf(image_path)
    encoded_img = caption_model.encoder(caption_model.cnn_model(img), training=False)
    
    decoded_caption = sos 
    for i in range(max_decoded_sentence_length)
        tokenized_caption = tokenizer([decoded_caption])[, -1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = caption_model.decoder(tokenized_caption, encoded_img, training=False, mask=mask)
        sampled_token_index = np.argmax(predictions[0, i, ])
        sampled_token = index_lookup[sampled_token_index]
        if sampled_token == eos
            break
        decoded_caption +=   + sampled_token

    return decoded_caption.replace(sos , )

# Streamlit App
def main()
    st.title(Image Captioning with Streamlit)
    st.write(Upload an image and the model will generate a caption for it.)

    uploaded_file = st.file_uploader(Choose an image..., type=[jpg, jpeg, png])
    if uploaded_file is not None
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_container_width=True)  # Updated to use_container_width
        st.write(Generating caption...)

        with open(temp_image.jpg, wb) as f
            f.write(uploaded_file.getbuffer())

        tokenizer = load_tokenizer()
        caption_model = load_model()

        caption = generate_caption(temp_image.jpg, caption_model, tokenizer)
        st.write(Generated Caption, caption)

if __name__ == __main__
    main()