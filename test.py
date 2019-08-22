import pandas as pd
import numpy as np
from keras.layers import Input, Conv2D
from keras.models import Model
from PIL import Image  
import matplotlib.pyplot as plt

input_file = ("/home/flipper/ananthsmap/req/27220156.csv") 
SMAP_LABLE='SoilMoisture'

latD=6
lonD=6
varD=1

inputs = Input((latD,lonD,varD))
  
df = pd.read_csv(input_file)
#filter bad val -9999

df=df[df[SMAP_LABLE]>-2000]

print(df)
#get Original output data
orig_x= df[SMAP_LABLE].values
print('1d shape')
print(orig_x.shape)

soilm=orig_x.reshape(-1, 6)

#soilm   = np.array(orig_x.reshape(latD,lonD))
print('2d shape')
print(soilm.shape)
print(soilm)

soilm = 255 * (1.0 - soilm)
soilm.resize((20,20))
im = Image.fromarray(soilm.astype(np.uint8), mode='L')
#im = im.resize((140, 140))
plt.imshow(im)
#im.show()
