# import required libraries
import requests
from bs4 import BeautifulSoup
import csv
import time
import os


# Base URLs and constants
BASE_URL = "https://comments.cftc.gov/PublicComments/CommentList.aspx?id=7512&ctl00_ctl00_cphContentMain_MainContent_gvCommentListChangePage={}_50"
COMMENT_BASE_URL = "https://comments.cftc.gov/PublicComments/ViewComment.aspx?id={}"
PAGE_COUNT = 17
CSV_FILE = "comments.csv"

def get_full_link(href):
	return f"https://comments.cftc.gov/PublicComments/{href}" if href else ""

def get_comment_details(comment_id):
	url = COMMENT_BASE_URL.format(comment_id)
	resp = requests.get(url)
	if resp.status_code != 200:
		return "", 0, "", ""
	soup = BeautifulSoup(resp.text, "html.parser")
	comment_text = ""
	div = soup.find("div", class_="ClearBoth", style=lambda s: s and "word-wrap: break-word" in s)
	if div:
		# Replace all <br> tags with '\n'
		for br in div.find_all("br"):
			br.replace_with("\n")
		# Get all text, preserving newlines
		comment_text = div.get_text(separator="\n", strip=True).replace('\xa0', ' ')
	# Remove "Comment Text:" prefix if present
	if comment_text.startswith("Comment Text:"):
		comment_text = comment_text[len("Comment Text:"):].lstrip()
	# Check for attachments
	has_attachments = 0
	attachment_link = ""
	attachment_filename = ""
	attachments_table = soup.find("table", class_="rgMasterTable", id=lambda x: x and "gvAttachments" in x)
	if attachments_table:
		row = attachments_table.find("tr", class_="rgRow")
		if row:
			link = row.find("a", href=True)
			if link and "PdfHandler.ashx?id=" in link["href"]:
				has_attachments = 1
				# Remove leading ../ if present
				href = link["href"].lstrip("../")
				attachment_link = "https://comments.cftc.gov/" + href
				attachment_filename = link.get_text(strip=True)
	return comment_text, has_attachments, attachment_link, attachment_filename

def parse_page(html):
	soup = BeautifulSoup(html, "html.parser")
	table = soup.find("table", {"class": "rgMasterTable"})
	if not table:
		return []
	rows = table.find_all("tr", class_=["rgRow", "rgAltRow"])
	data = []
	for row in rows:
		cols = row.find_all("td")
		if len(cols) < 6:
			continue
		# Extract fields
		date_received = cols[0].get_text(strip=True)
		release = cols[1].get_text(strip=True)
		first_name = cols[2].get_text(strip=True)
		last_name = cols[3].get_text(strip=True)
		organization = cols[4].get_text(strip=True)
		# Find the comment link and id
		link_tag = cols[5].find("a", href=True)
		comment_id = ""
		comment_link = ""
		comment_text = ""
		has_attachments = 0
		attachment_link = ""
		attachment_filename = ""
		if link_tag and "ViewComment.aspx?id=" in link_tag["href"]:
			comment_link = get_full_link(link_tag["href"])
			comment_id = link_tag["href"].split("id=")[-1]
			# Fetch comment details
			comment_text, has_attachments, attachment_link, attachment_filename = get_comment_details(comment_id)
		data.append({
			"id": comment_id,
			"Date Received": date_received,
			"Release": release,
			"First Name": first_name,
			"Last Name": last_name,
			"Organization": organization,
			"comment link": comment_link,
			"comment text": comment_text,
			"has attachments": has_attachments,
			"attachment link": attachment_link,
			"attachment filename": attachment_filename
		})
	return data

def get_headers(html):
	soup = BeautifulSoup(html, "html.parser")
	table = soup.find("table", {"class": "rgMasterTable"})
	if not table:
		return []
	header_row = table.find("tr")
	headers = ["id"]
	for th in header_row.find_all("th"):
		text = th.get_text(strip=True)
		if text == "Edit":
			headers.append("comment link")
		else:
			headers.append(text)
	# Add new fields
	headers.extend(["comment text", "has attachments", "attachment link", "attachment filename"])
	return headers


def download_attachment(url, save_dir):
	try:
		local_filename = None
		if hasattr(download_attachment, 'filename') and download_attachment.filename:
			local_filename = download_attachment.filename
		else:
			local_filename = url.split("/")[-1]
		local_path = os.path.join(save_dir, local_filename)
		resp = requests.get(url, stream=True)
		if resp.status_code == 200:
			with open(local_path, "wb") as f:
				for chunk in resp.iter_content(chunk_size=8192):
					f.write(chunk)
			print(f"Downloaded: {local_filename}")
		else:
			print(f"Failed to download {url}")
	except Exception as e:
		print(f"Error downloading {url}: {e}")

def main():
	all_data = []
	headers = None
	attachments_dir = os.path.join(os.path.dirname(__file__), "attachments")
	os.makedirs(attachments_dir, exist_ok=True)
	for page in range(1, PAGE_COUNT + 1):
		url = BASE_URL.format(page)
		print(f"Fetching page {page}...")
		resp = requests.get(url)
		if resp.status_code != 200:
			print(f"Failed to fetch page {page}")
			continue
		if headers is None:
			headers = get_headers(resp.text)
		page_data = parse_page(resp.text)
		all_data.extend(page_data)
		time.sleep(1)  # be polite to the server
	# Download attachments
	for row in all_data:
		if row.get("has attachments") and row.get("attachment link"):
			# Pass filename to download_attachment using function attribute
			download_attachment.filename = row.get("attachment filename")
			download_attachment(row["attachment link"], attachments_dir)
			download_attachment.filename = None
	# Write to CSV
	with open(CSV_FILE, "w", newline='', encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=headers)
		writer.writeheader()
		for row in all_data:
			writer.writerow(row)
	print(f"Saved {len(all_data)} rows to {CSV_FILE}")

if __name__ == "__main__":
	main()
