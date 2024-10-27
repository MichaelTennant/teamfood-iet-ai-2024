# https://docs.streamlit.io/develop/quick-reference/cheat-sheet

import streamlit as st
from pathlib import Path

PROJECT_ROOT=Path(__file__).parent

st.set_page_config(
    page_title='Team Food - AI IET Challenge',
    page_icon=':pizza:',    # This is an emoji shortcode. Could be a URL too.
)

'''
# :house: AI House Pricing Prediction (Team Food)
*We were here for the food.*

---

## Training AI to  Decect Pricing from Property Images
### Augmenting Image Data Decect Pricing from Property Images
To train a reliable image based AI model thousands of training images are required, however only 1712 images are supplied. Fortunately if you just apply a few transformations to the image, the AI won't be able to detect it is an image of the same place and you have effectively generated additional training data to create a more reliable model. This is otherwise known as image augmentation. 

For each image in the training data we randomly applied randomly applied a selection of the following transformations to generate 20 additional images.
 - Flip from left to right (training on upsidedown houses doesn't make much sense).
 - Rotate between 0-10 degrees clockwise or anticlockwise.
 - Stretch image in the x and y axis independently.
 - Gaussian blur with weight between 0.0 and 2.0 on image.
 - Shift image colour tempurature.
 - Shift image brightness.
 - Shift image saturation.

Randomly cropping images was experimented with - however due to concerns of loosing important details in the training data such as cropping out windows.

#### Original Image
'''
st.image(f"{PROJECT_ROOT}/res/img/augmentation_before.jpg")
'''
#### Augmented Image
'''
st.image(f"{PROJECT_ROOT}/res/img/augmentation_after.jpg")
'''
In this particular example, the image is horizontally blured, flipped, and the colour temperature and saturation is increased. This creating a distinct image according to the AI model.

After generating the augmented images, we now have 84 distinct training images for each property and a total of 35952 unique images to train the model. A much more respectable number than 1712.

### Counting Windows
We realized another metric we can look at in properties to estimate the price is the quantity of windows a property has. In 1696 England introduced a window tax increase in the value of properties with more windows. While California hasn't introduced a window tax, house pricing follows a similar trend to England in regards to window quantity.

By using resnet, we can annotate and count the number of windows visible in the frontal image.

### Collaging Images
We initially decided to create a collage of the matching bathroom, bedroom, frontal, and kitchen images to combine them into a single input image for each property to pass to the model. However, we quickly realized it was unnecessary and may also render undesirable results when combined with random image augmentation so scrapped the idea entirely. 

### Depth Perception
Using [DPT Large](https://huggingface.co/Intel/dpt-large), a pretrained depth prediction model created by professionals at intel, we can estimate the open area in the interior property photos. 

Open planned properties typically have a higher demand and thus price as opposed to closed-planned properties. This is necessary data as the property square foot area cannot necessarly be used to determine how open planned a property is.

#### DPT Large Input
'''
st.image(f"{PROJECT_ROOT}/res/img/dpt_before.jpg")
'''
#### DPT Large Output
'''
st.image(f"{PROJECT_ROOT}/res/img/dpt_after.png")
'''
### Image Based Price Prediction Model
 - image -> resize to 448x448 (3 colour channels) -> model
 - uses relu
---

## Training AI to Decect Pricing from Property Statistics
### Ranking Zip-Codes
We noticed that each zip-code in the test data was also in the training data so we replaced each zip-code with its average associated property prices. 

This allows for the AI model to use average zip-code prices as an input with a clear association instead of the seemingly random zip-codes themselves.

However it is plausible that AI model will encounter a zip-code it has never encountered before and thus does not have the average property price for the area stored. If this occurs the node will be passed the average property price from the training data overall.

### Staticics Based Price Prediction Model
Uses a multilayer perceptron

### Combining the models
Each AI model was trained separately...

To combine all the models, after the staticics based price prediction model is trained, we add the following addition numerical inputs and continue training
 - Lower Quartile Distance from DPT Large on bathroom.
 - Lower Quartile Distance from DPT Large on bedroom.
 - Lower Quartile Distance from DPT Large on kitchen.
 - Quantity of detected windows from frontal.
 - Detected price from image based price prediction model
'''