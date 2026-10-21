# Custom Domain & SSL Deployment Guide
**Target URL:** `https://your-classroom-agent.example.com`

This guide explains how to deploy the **Classroom Intelligence Agent** on an Ubuntu/Debian cloud server (AWS EC2, DigitalOcean, GCP Compute Engine, Linode, or any VPS) with a custom domain, Nginx reverse proxy, and free automated Let's Encrypt SSL.

---

## Step 1: DNS Setup (Domain Registrar)

Go to your DNS provider (Cloudflare, GoDaddy, Namecheap, Route 53, etc.) and add an **A Record**:

| Type | Host / Name | Value / Destination | TTL |
| :--- | :--- | :--- | :--- |
| **A** | `your-classroom-agent` | `<YOUR_SERVER_PUBLIC_IP>` | Auto / 300s |

*Example: If your domain is `example.com`, this creates `your-classroom-agent.example.com` pointing to your server's IP address.*

---

## Step 2: Clone & Set Up the Agent on the Server

SSH into your server:
```bash
ssh ubuntu@<YOUR_SERVER_PUBLIC_IP>
```

Install system dependencies:
```bash
sudo apt update && sudo apt install -y python3-venv python3-pip git nginx certbot python3-certbot-nginx libgl1 libglib2.0-0 libsndfile1 portaudio19-dev
```

Clone the repository:
```bash
git clone https://github.com/<your-username>/multimodal-attendance-agent.git
cd multimodal-attendance-agent
```

Set up Python virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Step 3: Configure Systemd Daemon (Auto-Start & Restart)

Copy the pre-configured systemd service:
```bash
sudo cp config/classroom_agent.service /etc/systemd/system/classroom_agent.service
```

*(Optional: Edit user or paths if your username is not `ubuntu`)*
```bash
sudo nano /etc/systemd/system/classroom_agent.service
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable classroom_agent
sudo systemctl start classroom_agent
```

Verify it is running:
```bash
sudo systemctl status classroom_agent
```

---

## Step 4: Configure Nginx & SSL

1. Copy the Nginx configuration:
```bash
sudo cp config/nginx_classroom_agent.conf /etc/nginx/sites-available/classroom_agent.conf
```

2. Replace `your-classroom-agent.example.com` with your real domain:
```bash
sudo sed -i 's/your-classroom-agent.example.com/<YOUR_ACTUAL_DOMAIN>/g' /etc/nginx/sites-available/classroom_agent.conf
```

3. Enable the site and test configuration:
```bash
sudo ln -s /etc/nginx/sites-available/classroom_agent.conf /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

4. Issue Free Automated SSL Certificate with Certbot:
```bash
sudo certbot --nginx -d <YOUR_ACTUAL_DOMAIN>
```

Certbot will automatically configure HTTPS, install the certificates, and renew them automatically before expiry.

---

## Step 5: Test Your Live Custom Domain

Open your browser and navigate to:
```
https://your-classroom-agent.example.com
```

- **HTTPS Secure Padlock 🔒**: Modern browsers will automatically grant camera and microphone permissions because the site is served over valid SSL.
- **WebSocket Streaming**: Continuous video and real-time telemetry will stream smoothly through Nginx.
- **Zero Cloud Leakage**: All OpenCV perception and acoustic VAD models continue executing 100% locally on the server.
