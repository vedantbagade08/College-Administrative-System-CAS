"""
Enhanced Camera Fallback - Professional Admin Dashboard with Dataset Face Recognition
Modern UI with DeepFace integration for student/faculty recognition
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import cv2
import threading
import os
from datetime import datetime
import requests
import base64
import numpy as np
from deepface import DeepFace

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

class ModernCameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CAS Admin Dashboard - Camera & Attendance System")
        
        # Window setup
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        w, h = min(1600, int(screen_w*0.9)), min(950, int(screen_h*0.9))
        self.root.geometry(f"{w}x{h}+{(screen_w-w)//2}+{(screen_h-h)//2}")
        
        # Modern colors
        self.c = {
            'bg': '#0a192f', 'sidebar': '#112240', 'card': '#1e3a5f',
            'accent': '#64ffda', 'danger': '#f43f5e', 'white': '#ccd6f6',
            'gray': '#8892b0', 'success': '#10b981'
        }
        self.root.configure(bg=self.c['bg'])
        
        # State
        self.API_BASE = 'http://localhost:5000'
        self.cap = None
        self.is_running = False
        self.camera_index = 0
        self.face_cascade = None
        self.dataset_path = os.path.join(os.path.dirname(__file__), 'datasets')
        self.detect_faces = True
        self.frame_count = 0
        self.frames_sent = 0
        self.face_count = 0
        self.known_students = []
        self.attendance_logs = []
        self.recognition_cache = {}  # Cache for recognized faces
        self.frame_skip = 0
        
        self.setup_ui()
        self.load_models()
        self.root.after(100, self.init_system)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def setup_ui(self):
        """Create modern UI"""
        # Header
        hdr = tk.Frame(self.root, bg=self.c['sidebar'], height=70)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        
        tk.Label(hdr, text="🎓 CAS Admin Dashboard", font=('Segoe UI', 22, 'bold'),
                bg=self.c['sidebar'], fg=self.c['white']).pack(side=tk.LEFT, padx=30, pady=15)
        
        self.header_stat = tk.Label(hdr, text="System Ready", font=('Segoe UI', 10),
                                    bg=self.c['sidebar'], fg=self.c['accent'])
        self.header_stat.pack(side=tk.RIGHT, padx=30)
        
        # Main container
        main = tk.Frame(self.root, bg=self.c['bg'])
        main.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Sidebar
        self.create_sidebar(main)
        
        # Content
        self.create_content(main)
        
        # Footer
        ftr = tk.Frame(self.root, bg=self.c['sidebar'], height=30)
        ftr.pack(fill=tk.X)
        ftr.pack_propagate(False)
        
        self.footer = tk.Label(ftr, text="🟢 Ready", font=('Segoe UI', 9),
                              bg=self.c['sidebar'], fg=self.c['gray'])
        self.footer.pack(side=tk.LEFT, padx=20)
    
    def create_sidebar(self, parent):
        """Create sidebar with controls"""
        sb = tk.Frame(parent, bg=self.c['sidebar'], width=360)
        sb.pack(side=tk.LEFT, fill=tk.Y, padx=(0,15))
        sb.pack_propagate(False)
        
        # Camera controls card
        self.card(sb, "🎥 Camera Controls", self.camera_controls)
        self.card(sb, "📊 Statistics", self.stats_display)
        self.card(sb, "🌐 Connection", self.conn_display)
        self.card(sb, "⚡ Actions", self.actions_display)
    
    def card(self, parent, title, content_fn):
        """Create styled card"""
        card = tk.Frame(parent, bg=self.c['card'])
        card.pack(fill=tk.X, padx=15, pady=10)
        
        # Title
        tk.Label(card, text=title, font=('Segoe UI', 11, 'bold'),
                bg=self.c['card'], fg=self.c['white']).pack(anchor=tk.W, padx=15, pady=10)
        
        # Content
        content = tk.Frame(card, bg=self.c['card'])
        content.pack(fill=tk.X, padx=15, pady=(0,15))
        content_fn(content)
    
    def camera_controls(self, p):
        """Camera control widgets"""
        tk.Label(p, text="Camera:", bg=self.c['card'], fg=self.c['gray'],
                font=('Segoe UI', 9)).pack(anchor=tk.W)
        
        self.cam_var = tk.StringVar(value="0")
        tk.Spinbox(p, from_=0, to=5, textvariable=self.cam_var, width=10,
                  bg=self.c['bg'], fg=self.c['white']).pack(anchor=tk.W, pady=5)
        
        self.start_btn = self.btn(p, "▶ Start Camera", self.start_cam, self.c['success'])
        self.stop_btn = self.btn(p, "⬛ Stop Camera", self.stop_cam, self.c['danger'], tk.DISABLED)
    

    
    def stats_display(self, p):
        """Statistics display"""
        self.stat_frame = self.info(p, "Frames:", "0")
        self.stat_faces = self.info(p, "Recognized:", "0")
        self.stat_sent = self.info(p, "Sent:", "0")
    
    def conn_display(self, p):
        """Connection status"""
        f = tk.Frame(p, bg=self.c['card'])
        f.pack(fill=tk.X)
        
        self.conn_dot = tk.Canvas(f, width=15, height=15, bg=self.c['card'], highlightthickness=0)
        self.conn_dot.pack(side=tk.LEFT)
        self.conn_dot.create_oval(2,2,13,13, fill='#ef4444', tags='dot')
        
        self.conn_lbl = tk.Label(f, text="Disconnected", font=('Segoe UI', 10, 'bold'),
                                bg=self.c['card'], fg='#ef4444')
        self.conn_lbl.pack(side=tk.LEFT, padx=10)
    
    def actions_display(self, p):
        """Action buttons"""
        self.btn(p, "🔄 Refresh Logs", self.refresh_logs, self.c['bg'])
        self.btn(p, "🔌 Test Server", self.test_conn, self.c['bg'])
        self.btn(p, "🔄 Clear Cache", lambda: setattr(self, 'recognition_cache', {}), self.c['accent'])
    
    def create_content(self, parent):
        """Create main content area"""
        cont = tk.Frame(parent, bg=self.c['bg'])
        cont.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Camera feed
        cam_frame = tk.Frame(cont, bg=self.c['card'])
        cam_frame.pack(fill=tk.BOTH, expand=True, pady=(0,15))
        
        tk.Label(cam_frame, text="📹 Live Camera Feed", font=('Segoe UI', 13, 'bold'),
                bg=self.c['card'], fg=self.c['white']).pack(anchor=tk.W, padx=20, pady=15)
        
        vid_container = tk.Frame(cam_frame, bg='#000000')
        vid_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0,15))
        
        self.video_lbl = tk.Label(vid_container, text="📹\n\nCamera Inactive\n\nStart camera to begin",
                                 bg='#000000', fg='#555555', font=('Segoe UI', 13))
        self.video_lbl.pack(expand=True, fill=tk.BOTH)
        
        # Logs panel
        log_frame = tk.Frame(cont, bg=self.c['card'], height=200)
        log_frame.pack(fill=tk.X)
        log_frame.pack_propagate(False)
        
        tk.Label(log_frame, text="📋 Attendance Logs", font=('Segoe UI', 13, 'bold'),
                bg=self.c['card'], fg=self.c['white']).pack(anchor=tk.W, padx=20, pady=15)
        
        self.logs_txt = scrolledtext.ScrolledText(log_frame, height=8, bg=self.c['bg'],
                                                  fg=self.c['white'], font=('Consolas', 9),
                                                  wrap=tk.WORD, relief=tk.FLAT)
        self.logs_txt.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0,15))
        self.logs_txt.insert('1.0', 'Waiting for logs...\n')
        self.logs_txt.config(state=tk.DISABLED)
    
    def btn(self, parent, text, cmd, bg, state=tk.NORMAL):
        """Create button"""
        b = tk.Button(parent, text=text, command=cmd, bg=bg, fg=self.c['white'],
                     font=('Segoe UI', 10, 'bold'), relief=tk.FLAT, padx=15, pady=8,
                     cursor='hand2', state=state)
        b.pack(fill=tk.X, pady=3)
        return b
    
    def info(self, parent, label, value):
        """Create info row"""
        f = tk.Frame(parent, bg=self.c['card'])
        f.pack(fill=tk.X, pady=2)
        
        tk.Label(f, text=label, font=('Segoe UI', 9), bg=self.c['card'],
                fg=self.c['gray']).pack(side=tk.LEFT)
        
        val_lbl = tk.Label(f, text=value, font=('Segoe UI', 9, 'bold'),
                          bg=self.c['card'], fg=self.c['white'])
        val_lbl.pack(side=tk.RIGHT)
        return val_lbl
    
    def load_models(self):
        """Load detection models"""
        try:
            face_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(face_path):
                self.face_cascade = cv2.CascadeClassifier(face_path)
                print(f"✓ Face cascade loaded")
            
            # Verify dataset path
            if os.path.exists(self.dataset_path):
                print(f"✓ Dataset found: {self.dataset_path}")
            else:
                print(f"⚠ Dataset not found: {self.dataset_path}")
        except Exception as e:
            print(f"Model load error: {e}")
    
    def init_system(self):
        """Initialize system"""
        self.test_conn()
        self.load_students()
        self.refresh_logs()
        threading.Thread(target=self.periodic_update, daemon=True).start()
    


    def test_conn(self):
        """Test server connection"""
        def test():
            try:
                r = requests.get(f"{self.API_BASE}/api/students", timeout=3)
                status = 'ok' if r.status_code == 200 else 'err'
            except:
                status = 'dis'
            self.root.after(0, lambda: self.update_conn_ui(status))
        threading.Thread(target=test, daemon=True).start()
    
    def update_conn_ui(self, status):
        """Update connection UI"""
        colors = {'ok': ('#22c55e', 'Connected'), 'err': ('#f59e0b', 'Error'),
                 'dis': ('#ef4444', 'Disconnected')}
        color, text = colors.get(status, colors['dis'])
        self.conn_dot.itemconfig('dot', fill=color)
        self.conn_lbl.config(text=text, fg=color)
    
    def load_students(self):
        """Load students"""
        def load():
            try:
                r = requests.get(f"{self.API_BASE}/api/students", timeout=3)
                if r.status_code == 200:
                    data = r.json()
                    self.known_students = [{'id': k, 'name': v.get('name', 'Unknown')}
                                          for k, v in data.get('students', {}).items()]
            except:
                pass
        threading.Thread(target=load, daemon=True).start()
    
    def refresh_logs(self):
        """Refresh attendance logs"""
        def fetch():
            try:
                r = requests.get(f"{self.API_BASE}/api/attendance/logs", timeout=3)
                if r.status_code == 200:
                    self.attendance_logs = r.json().get('logs', [])[:30]
                    self.root.after(0, self.update_logs_ui)
            except:
                pass
        threading.Thread(target=fetch, daemon=True).start()
    
    def update_logs_ui(self):
        """Update logs display"""
        self.logs_txt.config(state=tk.NORMAL)
        self.logs_txt.delete('1.0', tk.END)
        
        if not self.attendance_logs:
            self.logs_txt.insert('1.0', 'No logs available\n')
        else:
            for log in reversed(self.attendance_logs[-10:]):
                ts = log.get('timestamp', 'N/A')[:19]
                sid = log.get('id', 'N/A')[:12]
                name = log.get('name', 'Unknown')[:20]
                status = log.get('status', 'N/A')
                self.logs_txt.insert(tk.END, f"[{ts}] {sid} - {name} - {status}\n")
        
        self.logs_txt.config(state=tk.DISABLED)
    
    def periodic_update(self):
        """Periodic updates"""
        while True:
            self.test_conn()
            self.refresh_logs()
            threading.Event().wait(30)
    
    def start_cam(self):
        """Start camera"""
        try:
            idx = int(self.cam_var.get())
            self.cap = cv2.VideoCapture(idx)
            
            if not self.cap.isOpened():
                messagebox.showerror("Error", f"Cannot open camera {idx}")
                return
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            self.is_running = True
            self.frame_count = 0
            self.frames_sent = 0
            
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.header_stat.config(text="Camera Active", fg=self.c['success'])
            
            threading.Thread(target=self.capture_loop, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def stop_cam(self):
        """Stop camera"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.header_stat.config(text="Camera Stopped", fg=self.c['danger'])
        self.video_lbl.config(image='', text="Camera Inactive")
    
    def capture_loop(self):
        """Main capture loop with face recognition"""
        while self.is_running and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            self.frame_skip += 1
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = []
            
            if self.detect_faces and self.face_cascade:
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
                self.face_count = len(faces)
            
            # Process and recognize faces (every 10 frames to reduce load)
            recognized_faces = []
            if len(faces) > 0 and self.frame_skip % 10 == 0:
                for i, (x, y, w, h) in enumerate(faces):
                    face_img = frame[y:y+h, x:x+w]
                    name = self.recognize_face(face_img, i)
                    recognized_faces.append((x, y, w, h, name))
            else:
                # Use previous recognition or show generic
                recognized_faces = [(x, y, w, h, "Detecting...") for (x, y, w, h) in faces]
            
            # Draw rectangles and names
            for (x, y, w, h, name) in recognized_faces:
                # Green box for recognized, yellow for unknown
                color = (0, 255, 0) if name and name != "Unknown" and name != "Detecting..." else (0, 255, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Draw name with background
                label = name if name else "Unknown"
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x, y-text_h-10), (x+text_w+10, y), color, -1)
                cv2.putText(frame, label, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Stats overlay
            cv2.putText(frame, f"Frames: {self.frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Faces: {len(faces)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Send every 5th frame
            if self.frame_count % 5 == 0:
                threading.Thread(target=self.send_frame, args=(frame,), daemon=True).start()
            
            # Display
            self.root.after(0, self.update_display, frame)
    
    def recognize_face(self, face_img, face_idx):
        """Recognize face using DeepFace against dataset"""
        try:
            # Check cache first
            cache_key = f"face_{face_idx}_{self.frame_count // 30}"  # Cache for ~30 frames
            if cache_key in self.recognition_cache:
                return self.recognition_cache[cache_key]
            
            # Use DeepFace to find matching face in dataset
            # detector_backend='skip' prevents downloading detector models
            # enforce_detection=False allows processing without re-downloading
            result = DeepFace.find(
                img_path=face_img,
                db_path=self.dataset_path,
                model_name='Facenet512',
                detector_backend='skip',  # Skip detector to prevent downloads
                enforce_detection=False,
                silent=True
            )
            
            if result and len(result) > 0 and len(result[0]) > 0:
                # Get the matched image path
                matched_path = result[0]['identity'].iloc[0]
                
                # Extract name from path (e.g., datasets/Student/John_Doe/image.jpg -> John Doe)
                name_part = matched_path.split(os.sep)[-2]  # Get folder name
                name = name_part.replace('_', ' ').replace('-', ' ')
                
                # Cache the result
                self.recognition_cache[cache_key] = name
                return name
            
            return "Unknown"
        except Exception as e:
            print(f"Recognition error: {e}")
            return "Unknown"
    
    def send_frame(self, frame):
        """Send frame to server"""
        try:
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            b64 = base64.b64encode(buf).decode('utf-8')
            
            r = requests.post(f"{self.API_BASE}/api/camera/frame",
                            json={'image': f"data:image/jpeg;base64,{b64}",
                                 'timestamp': datetime.now().isoformat(),
                                 'frameNumber': self.frame_count},
                            timeout=1)
            
            if r.status_code == 200:
                self.frames_sent += 1
        except:
            pass
    
    def update_display(self, frame):
        """Update video display"""
        if not self.is_running:
            return
        
        try:
            h, w = frame.shape[:2]
            if w > 960 or h > 540:
                scale = min(960/w, 540/h)
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
            
            if PIL_AVAILABLE:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = ImageTk.PhotoImage(Image.fromarray(rgb))
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                _, buf = cv2.imencode('.png', rgb)
                img = tk.PhotoImage(data=buf.tobytes())
            
            self.video_lbl.config(image=img, text='')
            self.video_lbl.image = img
            
            # Update stats
            self.stat_frame.config(text=str(self.frame_count))
            self.stat_faces.config(text=str(self.face_count))
            self.stat_sent.config(text=str(self.frames_sent))
        except Exception as e:
            print(f"Display error: {e}")
    
    def on_close(self):
        """Handle window close"""
        self.stop_cam()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = ModernCameraApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()