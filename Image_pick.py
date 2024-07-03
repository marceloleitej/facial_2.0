import cv2
import random
import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Function to generate a random 6-digit ID
def generate_id(existing_ids):
    while True:
        new_id = random.randint(100000, 999999)
        if new_id not in existing_ids:
            return new_id

# Function to save the image
def save_image(image):
    output_dir = "Images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Retrieve existing IDs to avoid duplication
    existing_ids = set()
    for filename in os.listdir(output_dir):
        if filename.endswith(".png"):
            try:
                existing_ids.add(int(filename.split('.')[0]))
            except ValueError:
                continue

    image_id = generate_id(existing_ids)
    filename = os.path.join(output_dir, f"{image_id}.png")
    cv2.imwrite(filename, image)
    messagebox.showinfo("Success", f"Image captured and saved as {filename}")

# Function to show the preview of the captured image
def show_preview(cropped_frame):
    preview_window = tk.Toplevel()
    preview_window.title("Image Preview")

    # Convert the image to PIL format to display in Tkinter
    cv2image = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)

    panel = tk.Label(preview_window, image=imgtk)
    panel.imgtk = imgtk
    panel.pack(pady=10)

    # Function to confirm and save the image
    def confirm():
        save_image(cropped_frame)
        preview_window.destroy()

    confirm_button = tk.Button(preview_window, text="Confirm", command=confirm)
    confirm_button.pack(side=tk.LEFT, padx=20, pady=20)

    cancel_button = tk.Button(preview_window, text="Cancel", command=preview_window.destroy)
    cancel_button.pack(side=tk.RIGHT, padx=20, pady=20)

# Function to capture the image
def capture_image():
    global cap, rect_start, rect_end

    ret, frame = cap.read()
    if not ret:
        messagebox.showerror("Error", "Failed to capture image")
        return

    x_start, y_start = rect_start
    x_end, y_end = rect_end

    cropped_frame = frame[y_start:y_end, x_start:x_end]
    resized_frame = cv2.resize(cropped_frame, (216, 216))

    show_preview(resized_frame)

# Function to update the camera feed
def update_frame():
    global cap, panel, rect_start, rect_end

    ret, frame = cap.read()
    if not ret:
        return

    cv2.rectangle(frame, rect_start, rect_end, (0, 255, 0), 2)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)

    panel.imgtk = imgtk
    panel.config(image=imgtk)
    panel.after(10, update_frame)

# Function to create the GUI
def create_gui():
    global cap, panel, rect_start, rect_end

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Failed to access the webcam")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rect_width = 216
    rect_height = 216
    rect_start = (frame_width // 2 - rect_width // 2, frame_height // 2 - rect_height // 2)
    rect_end = (frame_width // 2 + rect_width // 2, frame_height // 2 + rect_height // 2)

    root = tk.Tk()
    root.title("Image Capture")

    # Create the panel to display the camera feed
    panel = tk.Label(root)
    panel.pack(pady=10)

    capture_button = tk.Button(root, text="Capture Image", command=capture_image)
    capture_button.pack(pady=20)

    update_frame()
    root.mainloop()

if __name__ == "__main__":
    create_gui()
