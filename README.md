# AI-Server: Surveillance & Theft Detection

## 📖 About the Project
This project is an advanced AI-powered surveillance and theft detection system. It leverages real-time video streaming, state-of-the-art object detection models (YOLO and RT-DETR), and a robust backend to monitor environments, identify potential theft, and track activities across different models and shop zones.

### Core Stack
- **Backend:** FastAPI (Python), utilizing `ultralytics` for YOLO models, OpenCV, ONNX Runtime (with AMD ROCm support), and SQLAlchemy for database ORM.
- **Frontend:** Nginx serving static HTML, CSS, and JS files, providing a dashboard for real-time monitoring and analytics.
- **Database:** PostgreSQL for storing surveillance logs, alerts, and system metadata.
- **Streaming:** MediaMTX (Bluenviron) handling RTSP and WebRTC video streams.
- **Deployment:** Fully containerized using Docker and Docker Compose.

---

## 🛠 Git Setup on a New Ubuntu PC (Start to Finish)

If you are setting up a brand new Ubuntu machine, follow these steps to install and configure Git so you can manage this repository.

### 1. Install Git
Open your terminal (`Ctrl + Alt + T`) and run:
```bash
sudo apt update
sudo apt install git -y
```

### 2. Configure Git
Set your name and email address. This will be attached to your commits.
```bash
git config --global user.name "Your Name"
git config --global user.email "your_email@example.com"
```
*(Optional)* Set the default branch name to `main`:
```bash
git config --global init.defaultBranch main
```

### 3. Generate an SSH Key
To securely clone and push to your remote repository (like GitHub/GitLab) without entering a password every time, generate an SSH key:
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```
Press **Enter** to accept the default file location, and optionally enter a passphrase.

### 4. Start the SSH Agent and Add Your Key
```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

### 5. Add the SSH Key to Your Git Provider (GitHub/GitLab)
Display your public key:
```bash
cat ~/.ssh/id_ed25519.pub
```
Copy the entire output.
- **GitHub:** Go to Settings -> SSH and GPG keys -> New SSH key. Paste your key and save.
- **GitLab:** Go to Preferences -> SSH Keys -> Add new key. Paste your key and save.

### 6. Clone the Repository
Now you can safely clone the project:
```bash
git clone git@github.com:YourUsername/YourProjectRepo.git
cd YourProjectRepo
```

---

## 🚀 How to Run the Project

This project uses **Docker** and **Docker Compose** to easily spin up all services (Frontend, Backend, Database, and MediaMTX) simultaneously. 

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) installed
- [Docker Compose](https://docs.docker.com/compose/install/) installed
- (Optional) NVIDIA or AMD GPUs configured for Docker if you are using GPU acceleration for AI models.

### Step-by-Step Execution

1. **Environment Variables Configuration**
   Ensure you have a `.env` file in the root directory (alongside `docker-compose.yml`). The file must contain the following variables:
   ```env
   POSTGRES_USER=your_db_user
   POSTGRES_PASSWORD=your_db_password
   POSTGRES_DB=surveillance
   DATABASE_URL=postgresql://your_db_user:your_db_password@db:5432/surveillance
   SECRET_KEY=your_super_secret_key
   HF_TOKEN=your_huggingface_token_if_needed
   ```

2. **Prepare AI Models**
   Ensure your model weights (`yolo11n.pt`, `yolo11s.pt`, `rtdetr-l.pt`, etc.) are placed inside the `backend/` directory.

3. **Start the Services**
   Run the following command from the root directory to build and start the containers in detached mode:
   ```bash
   sudo docker compose up -d --build
   ```

4. **Access the Application**
   Once the containers are up and running, you can access the various services at:
   - **Frontend (Dashboard):** [http://localhost:80](http://localhost:80)
   - **Backend (FastAPI Swagger UI):** [http://localhost:8000/docs](http://localhost:8000/docs)
   - **RTSP Stream:** `rtsp://localhost:8554`
   - **WebRTC Stream:** [http://localhost:8889](http://localhost:8889)

### Stopping the Services
To stop the application and safely bring down the containers, run:
```bash
sudo docker compose down
```

### Remote Deployment
If you want to deploy this stack to a remote server, we have provided a handy script:
```bash
./setup_remote.sh
```
This script will prompt you for SSH details, transfer the files via `rsync`, install Docker on the remote machine if missing, and automatically spin up the stack for you.
