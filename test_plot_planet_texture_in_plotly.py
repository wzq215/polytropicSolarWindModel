'''
import plotly.graph_objects as go
import numpy as np
from PIL import Image

def sphere(size, texture): 
    N_lat = int(texture.shape[0])
    N_lon = int(texture.shape[1])
    theta = np.linspace(0,2*np.pi,N_lat)
    phi = np.linspace(0,np.pi,N_lon)
    
    # Set up coordinates for points on the sphere
    x0 = size * np.outer(np.cos(theta),np.sin(phi))
    y0 = size * np.outer(np.sin(theta),np.sin(phi))
    z0 = size * np.outer(np.ones(N_lat),np.cos(phi))
    
    # Set up trace
    return x0,y0,z0

file_mars_texture_map = '/Users/jshept/Downloads/2k_mars.jpg'
texture = np.asarray(Image.open(file_mars_texture_map)).T

x,y,z = sphere(radius,texture)
surf = go.Surface(x=x, y=y, z=z,
                  surfacecolor=texture,
                  colorscale=colorscale)    

layout = go.Layout(scene=dict(aspectratio=dict(x=1, y=1, z=1)))

fig = go.Figure(data=[surf], layout=layout)

fig.show()
'''

import plotly.graph_objects as go
import numpy as np
import cv2
from sklearn.cluster import KMeans

# Function to remap image colors
def remap_image_colors(image_path, colorscale):
    # Read the image
    img = cv2.imread(image_path)

    # Normalize the image to 0-1
    img = img / 255.0

    # Convert the image back to 8-bit unsigned integers
    img = (img * 255).astype(np.uint8)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Map the grayscale values to the colorscale
    remapped_img = np.zeros_like(img)
    for i in range(3):
        remapped_img[:, :, i] = np.interp(gray, [0, 1], [colorscale[0][i], colorscale[-1][i]])

    return remapped_img

# Function to get colorscale
def get_colorscale(image_path, n_colors):
    # Read the image
    img = cv2.imread(image_path)

    # Reshape the image to be a list of RGB values
    reshaped_img = img.reshape(-1, 3)

    # Perform KMeans to find the most dominant colors
    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(reshaped_img)

    # The cluster centers are the dominant colors
    colorscale = kmeans.cluster_centers_

    # Normalize the colorscale to 0-1
    colorscale = colorscale / 255.0

    return colorscale.tolist()

# Generate the sphere
theta = np.linspace(0,2.*np.pi,100)
phi = np.linspace(0,np.pi,100)
x = np.outer(np.cos(theta),np.sin(phi))
y = np.outer(np.sin(theta),np.sin(phi))
z = np.outer(np.ones(np.size(theta)),np.cos(phi))

# Get the colorscale
image_path = '/Users/jshept/Downloads/2k_mars.jpeg'
colorscale = get_colorscale(image_path, 10)

# Remap the image colors
remapped_img = remap_image_colors(image_path, colorscale)

# Create the surface
surface = go.Surface(x=x, y=y, z=z, surfacecolor=remapped_img)

# Create the layout
layout = go.Layout(
    title='Mars Texture',
    scene=dict(
        xaxis=dict(gridcolor='rgb(255, 255, 255)', zerolinecolor='rgb(255, 255, 255)', showbackground=True, backgroundcolor='rgb(230, 230,230)'),
        yaxis=dict(gridcolor='rgb(255, 255, 255)', zerolinecolor='rgb(255, 255, 255)', showbackground=True, backgroundcolor='rgb(230, 230,230)'),
        zaxis=dict(gridcolor='rgb(255, 255, 255)', zerolinecolor='rgb(255, 255, 255)', showbackground=True, backgroundcolor='rgb(230, 230,230)'),
        aspectratio = dict(x=1, y=1, z=0.7),
        aspectmode = 'manual'
    )
)

# Create the figure
fig = go.Figure(data=[surface], layout=layout)

# Show the figure
fig.show()