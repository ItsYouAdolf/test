import cv2

v = 'gangster.mp4'


def face_capture(v):
    # путь к настройками(взять с github opencv) модель
    cascade_path = 'haarcascade_frontalface_default.xml'
    # захватывает видео
    video = cv2.VideoCapture(v)
    # обучение модель
    clf = cv2.CascadeClassifier(cascade_path)

    # как  не пытался сохранить в формате видео файла не выходило

    #size = (640, 480)
    #fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    #result = cv2.VideoWriter('filename.avi', fourcc, 20.0, size)

    # цикл в котором происходит классификация
    while True:
        # покадрово считает видеофайл
        _, frame = video.read()
        # переводит кадры в серый
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # классифицирует кадр с задыными настроиками
        faces = clf.detectMultiScale(
            # кадр
            gray,
            # Некоторые лица могут быть больше других, поскольку находятся ближе, чем остальные. Этот параметр компенсирует перспективу.
            scaleFactor=1.2,  #    scaleFactor=1.1            scaleFactor=1.2
            # Параметр minNeighbors определяет количество объектов вокруг лица. Слишком маленькое значение увеличит количество ложных срабатываний, а слишком большое сделает алгоритм более требовательным.
            minNeighbors=4,  #     minNeighbors=5             minNeighbors=6
            #размер этих областей.
            minSize=(115, 115),  # minSize=(100, 100)         minSize=(100, 100)
            # 3 настроики которые лучше всего себя показаль
        )

        for x, y, width, height in faces:
            # создает квадраты на кадрах(обводка лиц)
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)

        #result.write(frame)

        # открывает и показывает кадр уже с обводкой
        cv2.imshow('Faces', frame)
        # остоновить просмотр изображений
        if cv2.waitKey(1) == ord('q'):
            break
    #закрывает видео
    video.release()
    #result.release()
    # закрывает окно
    cv2.destroyAllWindows


def main(v):
    face_capture(v)

if __name__ == '__main__':
    main(v)
