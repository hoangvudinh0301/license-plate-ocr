import gradio as gr
from img import process_image
from cam import process_video

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
    title="Video License Plate Recobnition"
)

demo = gr.TabbedInterface(
    [image_interface, video_interface],
    ["Imgae", "Video"]
)
demo.launch()
