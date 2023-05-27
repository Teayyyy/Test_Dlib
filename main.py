import dlib
import cv2
import numpy

def main():
    # init
    detector = dlib.get_frontal_face_detector()
    model = '/Users/outianyi/Computer_Vision/shape_predictor_68_face_landmarks.dat'
    predictor = dlib.shape_predictor(model)
    capture = cv2.VideoCapture(1)
    if (capture.isOpened() is False):
        print("Camera Error!")

    while True:
        _, frame = capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)
        for face in faces:
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), [0, 255, 0], 1)
            cv2.imshow("Face detection with dlib", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


main()

