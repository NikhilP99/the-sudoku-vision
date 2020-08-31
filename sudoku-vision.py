from cv2 import cv2

video = cv2.VideoCapture(0)

print(video.get(cv2.CAP_PROP_FRAME_WIDTH))
print(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

#width
# video.set(3,1280)
#height
# video.set(4,720)

while(video.isOpened()):
    ret, frame = video.read()

    if ret == True:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        cv2.imshow("Video screen",gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

video.release()
cv2.destroyAllWindows()
