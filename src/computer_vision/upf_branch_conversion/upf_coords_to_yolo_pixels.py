import pandas as pd
import matplotlib.pyplot as plt


# This program use the Pandas library to read Excel files (which is what the UPF coords are stored in)
# and Matplotlib library to visualize the coordinates as pixels in an output file.

# Read Excel files
cello_body_df = pd.read_excel('cello_body.xlsx')
cello_fretboard_df = pd.read_excel('cello_fretboard.xlsx')
cello_antenna_front_df = pd.read_excel('cello_antenna_front.xlsx')
cello_antenna_back_df = pd.read_excel('cello_antenna_back.xlsx')
cellist_df = pd.read_excel('cellist.xlsx')

# Plotting
plt.figure(figsize=(8, 6))

# Plot cello body
plt.scatter(cello_body_df['x'], cello_body_df['y'], color='green', label='Cello Body')

# Plot cello fretboard
plt.scatter(cello_fretboard_df['x'], cello_fretboard_df['y'], color='red', label='Cello Fretboard')

# Plot cello antenna front
plt.scatter(cello_antenna_front_df['x'], cello_antenna_front_df['y'], color='gray', label='Cello Antenna front')

# Plot cello antenna back
plt.scatter(cello_antenna_back_df['x'], cello_antenna_back_df['y'], color='black', label='Cello Antenna back')

# Plot cellist
plt.scatter(cellist_df['x'], cellist_df['y'], color='blue', label='Cellist')

plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Cello and Cellist Coordinates')
plt.legend()
plt.grid(True)

#Save the image
plt.savefig('output.png')
plt.show()
