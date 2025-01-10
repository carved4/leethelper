from PIL import Image

# Create a 32x32 black image
img = Image.new("RGB", (32, 32), color="black")
img.save("static/favicon.ico")
