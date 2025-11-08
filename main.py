from ui.app_gui import *

def run_app():
    root = tb.Window(themename="darkly")
    NLPGuiApp(root)
    root.mainloop()

run_app()