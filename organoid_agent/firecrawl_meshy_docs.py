import os
from firecrawl import Firecrawl
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any
import time
import atexit
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize clients
firecrawl = Firecrawl(api_key=os.getenv('FIRECRAWL_API_KEY'))
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

# Cleanup function to close any active crawls
def cleanup_firecrawl():
    """Close any active Firecrawl browsers/crawls"""
    try:
        print("\nCleaning up Firecrawl resources...")
        active = firecrawl.get_active_crawls()
        if active and len(active) > 0:
            print(f"Found {len(active)} active crawls, canceling...")
            for crawl in active:
                try:
                    crawl_id = crawl.get('id') if isinstance(crawl, dict) else getattr(crawl, 'id', None)
                    if crawl_id:
                        firecrawl.cancel_crawl(crawl_id)
                        print(f"  Canceled crawl: {crawl_id}")
                except Exception as e:
                    print(f"  Error canceling crawl: {e}")
        else:
            print("No active crawls to cancel")
    except Exception as e:
        print(f"Cleanup error: {e}")

atexit.register(cleanup_firecrawl)

# Helper to convert Pydantic objects to dicts
def convert_to_dict(obj: Any) -> Dict:
    """Convert Pydantic Document object to dict"""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()
    elif hasattr(obj, 'dict'):
        return obj.dict()
    
    result = {}
    for attr in ['markdown', 'url', 'title', 'html', 'metadata']:
        if hasattr(obj, attr):
            value = getattr(obj, attr)
            if value is not None:
                result[attr] = value
    return result

# 1. Scrape documentation
def scrape_docs(url: str, use_crawl: bool = True) -> List[Dict]:
    """Scrape using Firecrawl V2 with cleanup"""
    
    cleanup_firecrawl()
    
    if use_crawl:
        print(f"Crawling: {url}")
        try:
            crawl_result = firecrawl.crawl(url=url, limit=50)
            
            if crawl_result:
                docs = []
                if isinstance(crawl_result, list):
                    for item in crawl_result:
                        docs.append(convert_to_dict(item))
                elif hasattr(crawl_result, 'data'):
                    for item in crawl_result.data:
                        docs.append(convert_to_dict(item))
                else:
                    docs.append(convert_to_dict(crawl_result))
                
                print(f"✓ Crawled {len(docs)} pages")
                return docs
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted! Cleaning up...")
            cleanup_firecrawl()
            raise
        except Exception as e:
            print(f"Crawl error: {e}")
            print("Falling back to single page...")
    
    print(f"Scraping single page: {url}")
    try:
        result = firecrawl.scrape(url=url, formats=['markdown'])
        if result:
            print(f"✓ Scraped 1 page")
            return [convert_to_dict(result)]
    except Exception as e:
        print(f"Scrape error: {e}")
        return []
    
    return []

# 2. Chunk documents
def chunk_document(doc: Dict, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """Split into overlapping chunks"""
    content = doc.get('markdown', '')
    if not content:
        return []
    
    chunks = []
    start = 0
    chunk_id = 0
    
    while start < len(content):
        end = start + chunk_size
        chunk_text = content[start:end]
        
        if chunk_text.strip():
            chunks.append({
                'text': chunk_text,
                'url': doc.get('url', ''),
                'title': doc.get('title', 'Untitled'),
                'chunk_id': chunk_id
            })
            chunk_id += 1
        
        start += (chunk_size - overlap)
    
    return chunks

# 3. Setup Pinecone with integrated inference
def setup_pinecone_index_with_inference(index_name: str = 'meshy-docs'):
    """Create index configured for integrated inference"""
    
    existing_indexes = [idx['name'] for idx in pc.list_indexes()]
    print(f"Existing indexes: {existing_indexes}")
    
    if index_name in existing_indexes:
        print(f"Deleting existing index to recreate with inference...")
        pc.delete_index(index_name)
        time.sleep(5)
    
    print(f"Creating index with integrated inference...")
    
    from pinecone import IndexEmbed, CloudProvider, AwsRegion
    
    index_config = pc.create_index_for_model(
        name=index_name,
        cloud=CloudProvider.AWS,
        region=AwsRegion.US_EAST_1,
        embed=IndexEmbed(
            model="multilingual-e5-large",
            field_map={"text": "chunk_text"},
            metric="cosine"
        )
    )
    
    print("Waiting for index to be ready...")
    time.sleep(10)
    
    return pc.Index(host=index_config.host)

# 4. Main pipeline
def process_and_store_docs(url: str = 'https://docs.meshy.ai/en', method: str = 'crawl'):
    """Complete pipeline with integrated inference"""
    
    try:
        print("="*80)
        print("STEP 1: SCRAPING")
        print("="*80)
        
        docs = scrape_docs(url, use_crawl=(method == 'crawl'))
        
        if not docs:
            print("\n⚠️  No documents scraped!")
            return
        
        print(f"\n✓ Scraped {len(docs)} page(s)")
        for i, doc in enumerate(docs[:3], 1):
            md_len = len(doc.get('markdown', ''))
            title = doc.get('title', 'No title')
            print(f"  Doc {i}: {md_len} chars - {title[:50] if title else 'No title'}")
        
        print("\n" + "="*80)
        print("STEP 2: PINECONE SETUP (with integrated inference)")
        print("="*80)
        
        try:
            index = setup_pinecone_index_with_inference('meshy-docs')
        except Exception as e:
            print(f"\n⚠️  Failed to setup index: {e}")
            import traceback
            traceback.print_exc()
            return
        
        print("\n" + "="*80)
        print("STEP 3: CHUNKING & UPLOADING (using upsert_records)")
        print("="*80)
        
        total_chunks = 0
        
        for doc_idx, doc in enumerate(docs):
            chunks = chunk_document(doc)
            
            if not chunks:
                print(f"Document {doc_idx + 1}: No chunks")
                continue
                
            print(f"Document {doc_idx + 1}: {len(chunks)} chunks")
            
            records = []
            for chunk in chunks:
                record_id = f"doc{doc_idx}_chunk{chunk['chunk_id']}"
                
                records.append({
                    "_id": record_id,
                    "chunk_text": chunk['text'],
                    "url": chunk['url'][:500] if chunk['url'] else '',
                    "title": chunk['title'][:200] if chunk['title'] else 'Untitled',
                    "chunk_id": chunk['chunk_id']
                })
            
            try:
                index.upsert_records(
                    namespace='docs',
                    records=records
                )
                total_chunks += len(records)
                print(f"  ✓ Uploaded {len(records)} records")
            except Exception as e:
                print(f"  ⚠️  Upload error: {e}")
                import traceback
                traceback.print_exc()
                return
        
        print(f"\n{'='*80}")
        print(f"✓ COMPLETE: {total_chunks} chunks from {len(docs)} documents stored")
        print(f"{'='*80}")
        
        stats = index.describe_index_stats()
        print(f"\nIndex stats: {stats}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Script interrupted by user")
        cleanup_firecrawl()
        raise
    except Exception as e:
        print(f"\n⚠️  Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_firecrawl()

# 5. Query using integrated inference - FIXED
def query_docs(question: str, index_name: str = 'meshy-docs', top_k: int = 3):
    """Query using integrated inference"""
    
    try:
        indexes = {idx['name']: idx for idx in pc.list_indexes()}
        if index_name not in indexes:
            print(f"Index '{index_name}' not found")
            return None
        
        index = pc.Index(host=indexes[index_name]['host'])
        
        # First, generate the embedding for the query using Pinecone inference
        from pinecone import EmbedModel
        
        query_embedding = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[question],
            parameters={
                "input_type": "query",
                "truncate": "END"
            }
        )
        
        # Extract the embedding vector
        query_vector = query_embedding.data[0].values
        
        # Now query with the vector
        results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            namespace='docs'
        )
        
        print(f"\n🔍 '{question}'")
        print("="*80)
        
        if not results or not results.get('matches'):
            print("No results found")
            return results
        
        for i, match in enumerate(results['matches'], 1):
            print(f"\n[{i}] Score: {match['score']:.4f}")
            print(f"    Title: {match['metadata'].get('title', 'N/A')}")
            print(f"    URL: {match['metadata'].get('url', 'N/A')}")
            # For regular query, metadata won't have chunk_text, just the preview
            chunk_text = match['metadata'].get('text', match['metadata'].get('chunk_text', ''))
            print(f"    {chunk_text[:200] if chunk_text else 'No text'}...")
        
        return results
        
    except Exception as e:
        print(f"Query error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("="*80)
    print("FIRECRAWL + PINECONE DOCUMENTATION INDEXER")
    print("="*80)
    print("\nCleaning up any existing Firecrawl sessions before starting...")
    cleanup_firecrawl()
    print()
    
    try:
        # Run pipeline
        process_and_store_docs('https://docs.meshy.ai/en', method='crawl')
        
        # Query
        existing = [idx['name'] for idx in pc.list_indexes()]
        if 'meshy-docs' in existing:
            print("\n" + "="*80)
            print("TESTING QUERIES")
            print("="*80)
            query_docs("How do I authenticate with the Meshy API?")
            query_docs("What are the rate limits?")
            query_docs("How do I generate 3D models?")
    except KeyboardInterrupt:
        print("\n\nExiting gracefully...")
    finally:
        print("\nFinal cleanup...")
        cleanup_firecrawl()
        print("Done!")
