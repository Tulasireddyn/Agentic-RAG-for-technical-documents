import arxiv
import datetime
import os
from tqdm import tqdm
import ssl
import time

# Fix SSL certificate verification issue on Windows
ssl._create_default_https_context = ssl._create_unverified_context

def fetch_papers(max_results=100):
    client = arxiv.Client(
        page_size=100,
        delay_seconds=3,
        num_retries=3
    )
    
    # Query for cs.AI
    search = arxiv.Search(
        query="cat:cs.AI",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    ninety_days_ago = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=90)
    
    if not os.path.exists("corpus"):
        os.makedirs("corpus")
        
    papers_metadata = []
    
    print(f"Fetching papers since {ninety_days_ago}...")
    
    results = list(client.results(search))
    print(f"Found {len(results)} potential papers.")
    
    for result in tqdm(results):
        if result.published < ninety_days_ago:
            continue
            
        papers_metadata.append({
            "title": result.title,
            "pdf_url": result.pdf_url,
            "published": result.published.isoformat(),
            "authors": [a.name for a in result.authors],
            "summary": result.summary,
            "id": result.entry_id
        })
        
        # Download PDF
        file_name = result.entry_id.split('/')[-1] + ".pdf"
        file_path = os.path.join("corpus", file_name)
        
        if not os.path.exists(file_path):
            try:
                print(f"Downloading {result.title}...")
                result.download_pdf(dirpath="corpus", filename=file_name)
                time.sleep(2) # Additional sleep
            except Exception as e:
                print(f"Error downloading {result.title}: {e}")
        
    print(f"Downloaded {len(papers_metadata)} papers.")
    return papers_metadata

if __name__ == "__main__":
    fetch_papers(max_results=10) 
