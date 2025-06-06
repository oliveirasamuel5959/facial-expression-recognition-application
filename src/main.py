import cv2

# Open the default camera
cam = cv2.VideoCapture(0)

# Detect face object haarcascade
detect_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

# Define rectangle positions
x0, y0 = 200, 300
x1, y1 = 300, 400

# Text formatting
font = cv2.FONT_HERSHEY_SIMPLEX

# Check if Camera was found
if not cam.isOpened():
    print("Error: Could not open video source.")
    exit()

while True:
    # get camera read boolean
    # and each camera frame
    ret, frame = cam.read()

    # detect face from frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = detect_face.detectMultiScale(gray_frame, 1.2, 3)
    
    # iterate over position and dimensions of the rectangle 
    # from cascade classifier
    for x, y, w, h in face:
        text_pos_x = x + (w // 3)
        text_pos_y = y + h + 20
        ret = cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
        cv2.putText(frame,'happy',(text_pos_x, text_pos_y), fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        # Write the frame to the output file
        out.write(frame)
        
    # Display the captured frame
    cv2.imshow('Camera', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()