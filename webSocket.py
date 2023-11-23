from ultralytics import YOLO
import asyncio
import websockets
import cv2
import json

async def stream_camera(websocket):
    await websocket.send(json.dumps({
        'opcode': 0,
        'msg': 'Hello World'
    }))
    await asyncio.sleep(0.5)
    # Load YOLO model
    model = YOLO("best.pt")
    vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    start_yolo = False
    if not vid.isOpened():
        print(f"Failed to open camera")
        return

    try:
        while vid.isOpened():
            ret, frame = vid.read()
            if not ret:
                print(f"Failed to read frame from camera")
                break

            keyName = cv2.waitKey(1)
            # 按下 q 結束
            if keyName == ord('q'):
                break

            # Run YOLO detection on the frame
            # results = model.track(frame, persist=True,conf=0.9)
            results = model.predict(frame, device="0")
            annotated_frame = results[0].plot()
            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            if not start_yolo:
                start_yolo = True
                await websocket.send(json.dumps({
                    'opcode': 1,
                    'msg': 'Camera starts'
                }))

            for result in results:                                         # iterate results
                boxes = result.boxes.cpu().numpy()                         # get boxes on cpu in numpy
                for box in boxes:                                          # iterate boxes
                    xy = box.xyxy[0].astype(int)                            # get corner points as int
                    name = result.names[int(box.cls[0])]
                    obj = {
                        'opcode': 2,
                        'msg': 'Predict',
                        'name': name,
                        'xy': xy.tolist()
                    }
                    print(obj)
                    await websocket.send(json.dumps(obj))

            await asyncio.sleep(0.1)
    except Exception as e:
        print(f"Exception while streaming camera: {e}")
        pass

    vid.release()
    cv2.destroyAllWindows()

start_server = websockets.serve(stream_camera, "127.0.0.1", 8000)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()