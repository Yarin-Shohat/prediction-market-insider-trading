# Data Collection

This folder is responsible for collecting, processing, and storing public comments from the CFTC (Commodity Futures Trading Commission) website regarding a specific rulemaking (ID 7512). It contains scripts and resources to automate the download of comment data and any associated attachments.

## Contents

- **main.py**: The primary script for scraping and downloading public comments and their attachments from the CFTC website.
- **comments.csv**: The output CSV file containing all collected comment data and metadata.
- **attachments/**: Directory where downloaded attachments (e.g., PDFs) are stored.
- **requirements.txt**: Lists the Python dependencies required to run the script.
- **Dockerfile**: (If present) Used to build a Docker image for running the data collection in a containerized environment.

## main.py Overview

- Scrapes all pages of public comments for the specified rulemaking from the CFTC website.
- For each comment, extracts metadata (date, name, organization, etc.), the full comment text, and checks for any attachments.
- Downloads any available attachments (e.g., PDF files) into the `attachments/` directory.
- Outputs all collected data into `comments.csv` with columns for all fields, including links and attachment info.
- Handles polite scraping (with delays) and basic error handling for failed requests.

## How to Run

### 1. Install Dependencies

It is recommended to use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Script

```bash
python main.py
```

- The script will fetch all comment data and attachments, saving them in `comments.csv` and the `attachments/` folder, respectively.
- Progress and download status will be printed to the console.

### 3. Output

- `comments.csv`: Contains all comment metadata, text, and attachment info.
- `attachments/`: Contains all downloaded files referenced in the comments.

## Docker Usage (Optional)

If a Dockerfile is present, you can build and run the scraper in a container:

```bash
docker build -t cftc-scraper .
docker run --rm -v $(pwd):/app cftc-scraper
```

## File Structure

```
Data Collection/
├── attachments/           # Downloaded attachments (PDFs, etc.)
├── comments.csv           # Output CSV with all comment data
├── main.py                # Main scraping script
├── requirements.txt       # Python dependencies
├── Dockerfile             # (Optional) Docker support
└── README.md              # This file
```

## Notes

- The script is designed to be polite to the CFTC server (1 second delay per page).
- If the scraping process is interrupted, you may need to delete partial files and rerun.
- The script can be adapted for other CFTC rulemakings by changing the `BASE_URL` and `PAGE_COUNT` constants in `main.py`.

## Requirements

- Python 3.7+
- See `requirements.txt` for required packages (e.g., `requests`, `beautifulsoup4`)