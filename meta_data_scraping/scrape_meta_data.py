# NOTE
# - right now look for matches of code on court listener and then just pick first one (heuristic)
# - get judge metadata in heuristic format, basically only if in cluster data for first match case, else not covererd (alternatively see hybrid script + check other hits)
# - author_judge = the person who wrote the majority opinion ; judge_names = the whole panel (everyone on stage)
# number observations len(cites) = 2076241
# number unique citations to run thorugh court listener len(set(cites)) = 409406
# -> 409406 / 4800 = 85ish , so would need about 85hours for all the data (but only need to consider test)
# - move this to py file


import os, time, pathlib
import requests, pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import sys
import json
from requests.exceptions import RequestException, Timeout, ReadTimeout

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from src.config import config
from src.data_prep_utils import build_data_suffix


load_dotenv()
API_TOKEN = os.getenv("CL_API_TOKEN")
if not API_TOKEN:
    raise SystemExit("Set CL_API_TOKEN in .env")


HEADERS = {
    "Authorization": f"Token {API_TOKEN}",
    "User-Agent":    "lepard-judge-enricher for research use (ADD MAIL HERE)",
    "Accept":        "application/json",
}

CITATION_LOOKUP_URL = "https://www.courtlistener.com/api/rest/v4/citation-lookup/"
MAX_RETRIES = 3
SLEEP = 0.75  # ~4800 req/h to stay under 5k/h limit (was 0.18!)
SAVE_INTERVAL = 1000  # Save progress every N citations
TIMEOUT = 45  # Increased timeout


def post_json(url, data):
    """Make POST request with robust error handling"""
    delay = SLEEP
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(SLEEP)
            r = requests.post(url, headers=HEADERS, data=data, timeout=TIMEOUT)
            if r.status_code == 429 or 500 <= r.status_code < 600:
                time.sleep(delay)
                delay *= 2
                continue
            r.raise_for_status()
            return r.json()
        except (Timeout, ReadTimeout) as e:
            print(f"Timeout on attempt {attempt + 1}/{MAX_RETRIES}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(delay)
                delay *= 2
                continue
        except RequestException as e:
            print(f"Request error on attempt {attempt + 1}/{MAX_RETRIES}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(delay)
                delay *= 2
                continue
        except Exception as e:
            print(f"Unexpected error in post_json: {e}")
            break
    return None


def get_json(url):
    """Make GET request with robust error handling"""
    delay = SLEEP
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(SLEEP)
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            if r.status_code == 429 or 500 <= r.status_code < 600:
                time.sleep(delay)
                delay *= 2
                continue
            r.raise_for_status()
            return r.json()
        except (Timeout, ReadTimeout) as e:
            print(f"Timeout on attempt {attempt + 1}/{MAX_RETRIES}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(delay)
                delay *= 2
                continue
        except RequestException as e:
            print(f"Request error on attempt {attempt + 1}/{MAX_RETRIES}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(delay)
                delay *= 2
                continue
        except Exception as e:
            print(f"Unexpected error in get_json: {e}")
            break
    return None


def enrich_one(cite: str) -> dict:
    """Enrich a single citation with metadata"""
    out = {
        "dest_cite": cite,
        "case_name": None,
        "court": None,
        "date_filed": None,
        "opinion_id": None,   # unused in cluster-only mode
        "cluster_id": None,
        "judge_names": [],
        "author_judge": None,
        "panel_text": None,
        "status": "not_found",
        "lookup_matches": 0,
    }
    if not cite:
        out["status"] = "error"
        return out

    try:
        payload = post_json(CITATION_LOOKUP_URL, {"text": cite.strip()})
        if not payload:
            out["status"] = "timeout_error"
            return out
        
        out["lookup_matches"] = len(payload)
        item = payload[0] if payload else {}

        # first cluster URL only
        cluster_url = None
        if isinstance(item.get("clusters"), list) and item["clusters"]:
            c = item["clusters"][0]
            cluster_url = c if isinstance(c, str) else (c.get("url") or c.get("resource_uri"))

        if cluster_url:
            cj = get_json(cluster_url)
            if cj:
                out["cluster_id"] = cj.get("id") or cj.get("pk")
                out["case_name"] = cj.get("case_name") or cj.get("caseName")
                court = cj.get("court")
                out["court"] = court.get("name") if isinstance(court, dict) else court
                out["date_filed"] = cj.get("date_filed") or cj.get("dateFiled")
                out["author_judge"] = cj.get("author_str") or cj.get("author")
                out["panel_text"] = cj.get("panel") or cj.get("panel_str")

                # judges (cluster-only)
                cj_judges = cj.get("judges")
                if cj_judges:
                    if isinstance(cj_judges, list):
                        out["judge_names"] = [n for n in cj_judges if n]
                    elif isinstance(cj_judges, str):
                        out["judge_names"] = [s.strip() for s in cj_judges.split(",") if s.strip()]

                out["status"] = "ok"
            else:
                out["status"] = "cluster_timeout_error"

    except Exception as e:
        print(f"Error processing citation '{cite}': {e}")
        out["status"] = "processing_error"

    return out


def load_cache(cache_path):
    """Load existing cache file if it exists"""
    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path)
            print(f"Loaded cache with {len(df)} entries from {cache_path}")
            return df
        except Exception as e:
            print(f"Error loading cache: {e}")
    return pd.DataFrame()


def save_progress(results_df, cache_path, final=False):
    """Save progress to cache file"""
    try:
        results_df.to_csv(cache_path, index=False)
        status = "Final results" if final else "Progress"
        print(f"{status} saved to {cache_path} ({len(results_df)} entries)")
    except Exception as e:
        print(f"Error saving progress: {e}")


def get_processed_cites(cache_df):
    """Get set of already processed citations"""
    if not cache_df.empty and 'dest_cite' in cache_df.columns:
        return set(cache_df['dest_cite'].astype(str))
    return set()


def main():
    INPUT_PATH = "finetuning_data_judge/test_not_in_scraped.csv"
    SAMPLE_N = None

    # derive a short tag from the input filename
    input_tag = pathlib.Path(INPUT_PATH).stem  # e.g., "test_dest_cite_2025_09_25"

    # Setup output directory and paths
    out_dir = os.path.join("meta_data_scraping", "meta_judge_data")
    os.makedirs(out_dir, exist_ok=True)

    data_suffix = build_data_suffix(config)
    sample_tag = "full" if SAMPLE_N is None else str(SAMPLE_N)

    cache_name = f"meta_judge_N_{input_tag}_{sample_tag}_{data_suffix}_cache.csv"
    final_name = f"meta_judge_N_{input_tag}_{sample_tag}_{data_suffix}.csv"
    cache_path = os.path.join(out_dir, cache_name)
    final_path = os.path.join(out_dir, final_name)

    # Load input data
    df = pd.read_csv(
        INPUT_PATH,
        compression="infer",
        usecols=["dest_cite"],
        dtype={"dest_cite": "string"}
    )
    
    # Choose sample and deduplicate
    if SAMPLE_N is None:
        cites = df["dest_cite"].dropna().astype(str).tolist()
    else:
        cites = df["dest_cite"].dropna().astype(str).head(SAMPLE_N).tolist()

    # Deduplicate while preserving order
    seen, unique_cites = set(), []
    for c in cites:
        if c not in seen:
            seen.add(c)
            unique_cites.append(c)

    print(f"Total unique citations to process: {len(unique_cites)}")

    # Load existing cache and determine what still needs processing
    cache_df = load_cache(cache_path)
    processed_cites = get_processed_cites(cache_df)
    
    remaining_cites = [c for c in unique_cites if c not in processed_cites]
    print(f"Already processed: {len(processed_cites)}")
    print(f"Remaining to process: {len(remaining_cites)}")

    if not remaining_cites:
        print("All citations already processed!")
        if not cache_df.empty:
            save_progress(cache_df, final_path, final=True)
            print(f"Final results saved to {final_path}")
        return

    # Process remaining citations with progress saving
    new_rows = []
    
    try:
        for i, cite in enumerate(tqdm(remaining_cites, desc="Enriching"), 1):
            result = enrich_one(cite)
            new_rows.append(result)
            
            # Save progress periodically
            if i % SAVE_INTERVAL == 0:
                current_df = pd.concat([cache_df, pd.DataFrame(new_rows)], ignore_index=True)
                save_progress(current_df, cache_path)
                print(f"Progress: {len(processed_cites) + i}/{len(unique_cites)} citations processed")
                
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving current progress...")
    except Exception as e:
        print(f"\nUnexpected error: {e}. Saving current progress...")

    # Combine all results and save final output
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        final_df = pd.concat([cache_df, new_df], ignore_index=True)
    else:
        final_df = cache_df

    # Ensure consistent column order
    cols = [
        "dest_cite","case_name","court","date_filed",
        "opinion_id","cluster_id","author_judge",
        "judge_names","panel_text","status","lookup_matches"
    ]
    final_df = final_df.reindex(columns=cols)

    # Save final results
    save_progress(final_df, cache_path)
    save_progress(final_df, final_path, final=True)

    # Print summary
    print(f"\nFinal Summary:")
    print(f"Total citations processed: {len(final_df)}")
    if 'status' in final_df.columns:
        print("Status breakdown:")
        print(final_df['status'].value_counts())


if __name__ == "__main__":
    main()