import tkinter as tk
from tkinter import ttk, filedialog, messagebox

APP_NAME = "TinyTool"

def open_file():
    path = filedialog.askopenfilename()
    if path:
        messagebox.showinfo(APP_NAME, f"Selected:\n{path}")

root = tk.Tk()
root.title(APP_NAME)
root.geometry("360x220")

# Use ttk for a cleaner look
style = ttk.Style()
# On macOS, 'aqua' theme is typical; fallback to default if unavailable
if "aqua" in style.theme_names():
    style.theme_use("aqua")

frm = ttk.Frame(root, padding=16)
frm.pack(fill="both", expand=True)

ttk.Label(frm, text="Hello, macOS 👋").pack(pady=8)
ttk.Button(frm, text="Open file…", command=open_file).pack()

# Native-ish menu bar with Cmd+Q
menubar = tk.Menu(root)
app_menu = tk.Menu(menubar, name="apple")   # macOS app menu slot
app_menu.add_command(label=f"About {APP_NAME}", command=lambda: messagebox.showinfo("About", APP_NAME))
menubar.add_cascade(menu=app_menu)
file_menu = tk.Menu(menubar, tearoff=False)
file_menu.add_command(label="Open…", command=open_file, accelerator="⌘O")
file_menu.add_separator()
file_menu.add_command(label="Quit", command=root.quit, accelerator="⌘Q")
menubar.add_cascade(label="File", menu=file_menu)
root.config(menu=menubar)

# Bind common shortcuts
root.bind_all("<Command-o>", lambda e: open_file())
root.bind_all("<Command-q>", lambda e: root.quit())

root.mainloop()
