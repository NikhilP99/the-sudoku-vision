import argparse
from cv2 import cv2

# setting up parser
parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='', help='Input file (eg: testpic.jpg)')
args = parser.parse_args()

def video_mode():
    video = cv2.VideoCapture(0)

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


def main():

    # if the user has not specified a file to load, use the video input
    if args.file == '':
        video_mode()
    else:
        image = cv2.imread(args.file)
        cv2.imshow("image",image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()