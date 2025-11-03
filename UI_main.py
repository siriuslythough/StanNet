import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from PIL import Image, ImageTk
import torch
import threading

# Import your utilities and validation functions
from utils import ToHSV, ToComplex
from validation import (
    build_model,
    build_transform,
    load_model_from_path,
    predict_single_image,
    find_saved_models
)

class YogaPoseClassifierUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Yoga Pose Classifier")
        self.root.geometry("900x750")
        self.root.resizable(True, True)
        
        # Variables
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.classes = None
        self.image_size = 224
        self.selected_image_path = None
        self.selected_model_path = None
        self.model_dir = "./saved_models"
        self.is_predicting = False
        
        # Create UI elements
        self.create_widgets()
        self.load_available_models()
        
    def create_widgets(self):
        """Create all UI components"""
        
        # Main frame with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # ===== Model Selection Section =====
        model_frame = ttk.LabelFrame(main_frame, text="Model Selection", padding="10")
        model_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(model_frame, text="Model Directory:").pack(side=tk.LEFT, padx=5)
        self.model_dir_entry = ttk.Entry(model_frame, width=40)
        self.model_dir_entry.insert(0, self.model_dir)
        self.model_dir_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        ttk.Button(model_frame, text="Browse", command=self.browse_model_dir).pack(side=tk.LEFT, padx=5)
        ttk.Button(model_frame, text="Refresh", command=self.load_available_models).pack(side=tk.LEFT, padx=5)
        
        # Available models dropdown
        model_select_frame = ttk.Frame(main_frame)
        model_select_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(model_select_frame, text="Select Model:").pack(side=tk.LEFT, padx=5)
        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(
            model_select_frame,
            textvariable=self.model_var,
            state="readonly",
            width=50
        )
        self.model_dropdown.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.model_dropdown.bind("<<ComboboxSelected>>", self.on_model_selected)
        
        ttk.Button(model_select_frame, text="Load Model", command=self.load_selected_model).pack(side=tk.LEFT, padx=5)
        
        # Model info label
        self.model_info_label = ttk.Label(main_frame, text="No model loaded", foreground="gray")
        self.model_info_label.pack(fill=tk.X, padx=5, pady=5)
        
        # ===== Image Selection Section =====
        image_frame = ttk.LabelFrame(main_frame, text="Image Selection", padding="10")
        image_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(image_frame, text="Browse Image", command=self.browse_image).pack(side=tk.LEFT, padx=5)
        self.image_path_label = ttk.Label(image_frame, text="No image selected", foreground="gray")
        self.image_path_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # ===== Image Preview Section =====
        preview_frame = ttk.LabelFrame(main_frame, text="Image Preview", padding="10")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.image_label = ttk.Label(preview_frame, text="Image preview will appear here", foreground="gray")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # ===== Prediction Section =====
        predict_frame = ttk.Frame(main_frame)
        predict_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(predict_frame, text="Predict", command=self.run_prediction).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(predict_frame, text="Top-K:").pack(side=tk.LEFT, padx=5)
        self.topk_var = tk.StringVar(value="5")
        topk_spinbox = ttk.Spinbox(
            predict_frame,
            from_=1,
            to=20,
            textvariable=self.topk_var,
            width=5
        )
        topk_spinbox.pack(side=tk.LEFT, padx=5)
        
        self.predict_status = ttk.Label(predict_frame, text="Ready", foreground="blue")
        self.predict_status.pack(side=tk.LEFT, padx=20)
        
        # ===== Results Section =====
        results_frame = ttk.LabelFrame(main_frame, text="Predictions", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create treeview for results
        columns = ("Rank", "Class", "Confidence (%)")
        self.results_tree = ttk.Treeview(results_frame, columns=columns, height=8, show="headings")
        
        self.results_tree.column("Rank", width=50, anchor=tk.CENTER)
        self.results_tree.column("Class", width=300, anchor=tk.W)
        self.results_tree.column("Confidence (%)", width=100, anchor=tk.CENTER)
        
        self.results_tree.heading("Rank", text="Rank")
        self.results_tree.heading("Class", text="Class")
        self.results_tree.heading("Confidence (%)", text="Confidence (%)")
        
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscroll=scrollbar.set)
        
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def browse_model_dir(self):
        """Browse and select model directory"""
        directory = filedialog.askdirectory(title="Select Model Directory", initialdir=self.model_dir)
        if directory:
            self.model_dir = directory
            self.model_dir_entry.delete(0, tk.END)
            self.model_dir_entry.insert(0, self.model_dir)
            self.load_available_models()
    
    def load_available_models(self):
        """Load and display available models from directory"""
        self.model_dir = self.model_dir_entry.get()
        try:
            model_files = find_saved_models(self.model_dir)
            model_names = [Path(f).name for f in model_files]
            self.model_dropdown["values"] = model_names
            
            if model_names:
                self.model_info_label.config(
                    text=f"Found {len(model_names)} model(s)",
                    foreground="green"
                )
            else:
                self.model_info_label.config(
                    text="No models found in directory",
                    foreground="orange"
                )
        except FileNotFoundError as e:
            self.model_info_label.config(text=str(e), foreground="red")
    
    def on_model_selected(self, event=None):
        """Called when a model is selected from dropdown"""
        selected = self.model_var.get()
        if selected:
            self.selected_model_path = Path(self.model_dir) / selected
    
    def load_selected_model(self):
        """Load the selected model"""
        if not self.selected_model_path or not Path(self.selected_model_path).exists():
            messagebox.showerror("Error", "Please select a valid model file")
            return
        
        try:
            self.predict_status.config(text="Loading model...", foreground="blue")
            self.root.update()
            
            self.model, self.classes, self.image_size = load_model_from_path(
                str(self.selected_model_path),
                self.device
            )
            
            num_classes = len(self.classes) if self.classes else "Unknown"
            self.model_info_label.config(
                text=f"✓ Model loaded: {Path(self.selected_model_path).name} | Classes: {num_classes} | Image Size: {self.image_size}",
                foreground="green"
            )
            self.predict_status.config(text="Ready", foreground="green")
            
        except Exception as e:
            self.model_info_label.config(text=f"Error loading model: {str(e)}", foreground="red")
            self.predict_status.config(text="Error", foreground="red")
            messagebox.showerror("Model Loading Error", str(e))
    
    def browse_image(self):
        """Browse and select an image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.selected_image_path = file_path
            self.image_path_label.config(text=Path(file_path).name, foreground="black")
            self.display_image_preview(file_path)
    
    def display_image_preview(self, image_path):
        """Display image preview in the UI"""
        try:
            img = Image.open(image_path)
            
            # Resize for display (max 300x300)
            display_size = (300, 300)
            img.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # Update label
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep a reference
            
        except Exception as e:
            self.image_label.config(text=f"Error loading preview: {str(e)}")
    
    def run_prediction(self):
        """Run prediction in a separate thread"""
        if not self.model:
            messagebox.showerror("Error", "Please load a model first")
            return
        
        if not self.selected_image_path:
            messagebox.showerror("Error", "Please select an image first")
            return
        
        # Run prediction in separate thread to avoid freezing UI
        thread = threading.Thread(target=self.predict_worker)
        thread.daemon = True
        thread.start()
    
    def predict_worker(self):
        """Worker function to run prediction"""
        try:
            self.predict_status.config(text="Predicting...", foreground="orange")
            self.root.update()
            
            topk = int(self.topk_var.get())
            
            vals, idxs, classes = predict_single_image(
                self.model,
                self.selected_image_path,
                self.image_size,
                self.device,
                self.classes,
                topk=topk
            )
            
            # Clear previous results
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)
            
            # Populate results
            for rank, (prob, class_idx) in enumerate(zip(vals, idxs), 1):
                class_name = classes[class_idx] if classes and class_idx < len(classes) else f"Class {class_idx}"
                confidence = prob * 100
                self.results_tree.insert("", tk.END, values=(rank, class_name, f"{confidence:.2f}"))
            
            self.predict_status.config(text="✓ Prediction complete", foreground="green")
            
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
            self.predict_status.config(text="Error", foreground="red")

def main():
    root = tk.Tk()
    app = YogaPoseClassifierUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
