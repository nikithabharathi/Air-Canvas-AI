#Air-Canvas-AI
A real-time AI-powered air writing system built using Python, OpenCV, and MediaPipe Hand Landmarker.

VisionInk allows users to draw in the air using hand gestures captured through a webcam. The project uses computer vision and real-time hand tracking to create a virtual drawing experience without touching the screen.

---

## 🚀 Features

- Real-time hand tracking
- Smooth air writing
- Gesture-based controls
- Virtual drawing canvas
- Hand landmark visualization
- Canvas clearing gesture
- Motion smoothing for cleaner strokes

---

## ✋ Gesture Controls

| Gesture | Action |
|---|---|
| ☝️ Index Finger | Draw |
| ✌️ Index + Middle Finger | Move without drawing |
| ✊ Fist | Clear canvas |

---

## 🧠 Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy

---

## 📂 Project Structure

```bash
VisionInk/
│
├── main.py
├── models/
├── src/
│   ├── hand_tracker.py
│   └── drawing_utils.py
│
├── utils/
│   └── constants.py
│
└── requirements.txt
