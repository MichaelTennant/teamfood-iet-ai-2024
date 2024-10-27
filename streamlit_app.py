# https://docs.streamlit.io/develop/quick-reference/cheat-sheet

import streamlit as st
from pathlib import Path
import pandas as pd


PROJECT_ROOT=Path(__file__).parent


st.set_page_config(
    page_title='Team Food - AI IET Challenge',
    page_icon=':pizza:',    # This is an emoji shortcode. Could be a URL too.
)

# Allow for page to be wider if nessessary (up to 1024 pixels)
st.markdown(
    """
    <style>
        .stMainBlockContainer {
            max-width: 1024px;
        }
    </style>
    """, 
    unsafe_allow_html=True
)

'''
---
# :house: AI House Price Prediction (Team Food)
*We were here for the food.*

---

## Training AI to  Detect Pricing from Property Images
### Augmenting Image Data Detect Pricing from Property Images
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
'''

col1, col2 = st.columns(2)

col1.header("Original Image")
col1.image(f"{PROJECT_ROOT}/res/img/augmentation_before.jpg")

col2.header("Augmented Image")
col2.image(f"{PROJECT_ROOT}/res/img/augmentation_after.jpg")

'''
In this particular example, the image is horizontally blured, flipped, and the colour temperature and saturation is increased. This creating a distinct image according to the AI model.

After generating the augmented images, we now have 84 distinct training images for each property and a total of 35952 unique images to train the model. A much more respectable number than 1712.

### Counting Windows
We realized another metric we can look at in properties to estimate the price is the quantity of windows a property has. In 1696 England introduced a window tax increase in the value of properties with more windows. While California hasn't introduced a window tax, house pricing follows a similar trend to England in regards to window quantity.

By using a pretrained [resnet](https://pytorch.org/hub/pytorch_vision_resnet/) model, we can annotate and count the number of windows visible in the frontal image.

### Collaging Images
We initially decided to create a collage of the matching bathroom, bedroom, frontal, and kitchen images to combine them into a single input image for each property to pass to the model. However, we quickly realized it was unnecessary and may also render undesirable results when combined with random image augmentation so scrapped the idea entirely. 
'''

st.image(f"{PROJECT_ROOT}/res/img/collage_graph.png")

'''
### Depth Perception
Using [DPT Large](https://huggingface.co/Intel/dpt-large), a pretrained depth prediction model created by professionals at intel, we can estimate the open area in the interior property photos. 

Open planned properties typically have a higher demand and thus price as opposed to closed-planned properties. This is necessary data as the property square ft area cannot necessarly be used to determine how open planned a property is.

Unfortunately the model does have a few issues and is not 100% reliable. For instance "99.kitchen.jpg" in image training data contains a large black oven, which DPT Large assumes is a long corridor.  
'''

col1, col2 = st.columns(2)

col1.header("DPT Large Input")
col1.image(f"{PROJECT_ROOT}/res/img/dpt_before.jpg")

col2.header("DPT Large Output")
col2.image(f"{PROJECT_ROOT}/res/img/dpt_after.png")

'''
### Image Based Price Prediction Model
The model is passed the image, which is resized to 448x448 with 3 colour channels. This data is further randomly augmented using keras by changing the contrast, flipping it, and rotating it. This is passed through several relu keras layers, before finally returning a single price prediction.
'''

st.image(f"{PROJECT_ROOT}/res/img/img_model_graph.png")

'''
Unfortunately due to time constraints we were unable to implement batch learning so we're limited to the 16GB GPU memory, and thus could only train the model on ~20% of the augmented data.

Error ~ 10%

---

## Training AI to Detect Pricing from Property Statistics
### Average Property Price in each Zip-Code
We noticed that each zip-code in the test data was also in the training data so we replaced each zip-code with its average associated property prices. 
'''

zipcode_avrprice_df = pd.read_csv(f"{PROJECT_ROOT}/res/data/zipcode_avrprice.csv").set_index("ZipCode")[["AvrPrice"]]
st.bar_chart(zipcode_avrprice_df, x_label="Zip-Code", y_label="Average Property Price ($USD)")

'''
This allows for the AI model to use average zip-code prices as an input with a clear association instead of the seemingly random zip-codes themselves.

However it is plausible that AI model will encounter a zip-code it has never encountered before and thus does not have the average property price for the area stored. If this occurs the node will be passed the average property price of ~$598598 from the training data overall.

### Staticics Based Price Prediction Model
Uses a multilayer perceptron with the following inputs
 - Bathroom Count.
 - Bedroom Count.
 - Square ft Area.
 - Average Price in Zip-Code (Or total average price if zip-code not in training data).

### Synthetic generated data
The synthetic data generation works by taking both the upper and lower quartiles of the housing data, and generating new data present within one standard deviation of each quartile respectively. This way, the generated data combined with the original data still has the same median and mean, whilst being able to have a higher variance, allowing for a more diverse set of data for the model to train from
this improved our root mean square error model results from 1.75 to 0.97.

### Combining the models
Each AI model was trained separately to prevent overcomplicating the model and speed up training time signficiantly.

To combine all the models, after the staticics based price prediction model is trained, we add the following addition numerical inputs and resume training it with the additional data
 - Lower Quartile Distance from DPT Large on bathroom.
 - Lower Quartile Distance from DPT Large on bedroom.
 - Lower Quartile Distance from DPT Large on kitchen.
 - Quantity of detected windows from frontal.

We then planned to use another AI model to combine the two results from the modified statistics based prediction model and the image based model, yielding a single output. However we ran out of time on training the image and image+stats combinator model.
'''
st.header("AI Model Pipeline Plan")
st.image(f"{PROJECT_ROOT}/res/img/combined_model_diagram.png")
'''
---
'''

col1, col2 = st.columns(2)

col1.header("Thank you!")
col1.image(f"{PROJECT_ROOT}/res/img/team_plus_extras.jpg")

col2.header("")
col2.image(f"{PROJECT_ROOT}/res/img/team_plus_extras_depth.png")

'''
---
'''