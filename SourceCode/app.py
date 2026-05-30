import gradio as gr
from img import process_image
from video import process_video

image_interface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="numpy"),
    outputs=[
        gr.Image(type="numpy"),
        gr.Textbox(label="Plate Number")
    ],
    title="Vietnam License Plate Recognition",
    description="Upload vehicle image"
)
video_interface = gr.Interface(
    fn=process_video,
    inputs=gr.Video(),
    outputs=gr.Video(),
    title="Video License Plate Recognition"
)

webcam_interface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(sources="webcam", type="numpy", streaming=True),
    outputs=[
        gr.Image(type="numpy"),
        gr.Textbox(label="Plate Number")
    ],
    live=True,
    title="Realtime Camera ANPR"
)

demo = gr.TabbedInterface(
    [image_interface, video_interface, webcam_interface],
    ["Imgae", "Video", "Webcam Live"]
)
demo.launch()
