import cv2

def crop_face(frame, pos, dim):
    # pos = [x, y]
    # dim = [w, h]
    faces = frame[pos[1]:pos[1] + dim[1], pos[0]: pos[0] + dim[0]]
    return faces