"""
Final script to collect automotive questions from multiple sources:
1. mechanics.stackexchange.com
2. Reddit r/MechanicAdvice
"""

import json
import time
from datetime import datetime
from stackapi import StackAPI
import os
import sys
import re
import praw
from typing import List, Dict, Optional

def get_available_tags(site, min_count=50):
    """Get popular tags from the site to find better alternatives"""
    try:
        print("Fetching available tags from the site...")
        tags = site.fetch('tags', sort='popular', pagesize=100)
        
        if 'items' in tags:
            # Filter for automotive-related tags with decent question count
            automotive_keywords = ['engine', 'fuel', 'electric', 'brake', 'transmission', 
                                  'battery', 'oil', 'coolant', 'exhaust', 'turbo',
                                  'clutch', 'suspension', 'tire', 'steering', 'air']
            
            relevant_tags = []
            for tag in tags['items']:
                name = tag.get('name', '')
                count = tag.get('count', 0)
                
                # Check if tag contains automotive keywords and has enough questions
                if count >= min_count:
                    for keyword in automotive_keywords:
                        if keyword in name.lower():
                            relevant_tags.append({
                                'name': name,
                                'count': count
                            })
                            break
            
            return relevant_tags
        return []
    except Exception as e:
        print(f"Error fetching tags: {e}")
        return []

def collect_automotive_questions_with_answers(num_per_tag=100):
    """
    Collect automotive questions with their accepted answers
    """
    
    # Initialize StackAPI
    site = StackAPI('mechanics')
    site.page_size = 100  # Max allowed
    site.max_pages = 5  # Allow multiple pages if needed
    
    # First, let's see what tags are actually available
    available_tags = get_available_tags(site)
    
    if available_tags:
        print("\nAvailable automotive-related tags:")
        for tag in available_tags[:15]:  # Show top 15
            print(f"  - {tag['name']}: {tag['count']} questions")
    
    # Use a mix of specific and general tags that actually exist
    # Based on common automotive topics
    tags = [
        'engine',           # Engine-related issues
        'electrical',       # Electrical systems
        'brakes',          # Braking system
        'transmission',    # Transmission issues
        'battery',         # Battery problems
        'oil',            # Oil-related questions
        'coolant',        # Cooling system
        'exhaust',        # Exhaust system
        'air-conditioning' # AC system
    ]
    
    # Storage
    all_questions = []
    
    # Statistics
    stats = {
        'total_questions': 0,
        'questions_with_accepted_answers': 0,
        'questions_with_answers_fetched': 0,
        'questions_by_tag': {},
        'collection_timestamp': datetime.now().isoformat(),
        'target_per_tag': num_per_tag,
        'actual_tags_used': [],
        'errors': []
    }
    
    print(f"\n{'='*60}")
    print(f"Starting collection: {stats['collection_timestamp']}")
    print(f"Target: {num_per_tag} questions per tag")
    print(f"Tags to try: {tags}")
    print(f"{'='*60}")
    
    for tag_index, tag in enumerate(tags, 1):
        print(f"\n[{tag_index}/{len(tags)}] Processing tag: '{tag}'")
        
        try:
            # Try to fetch questions
            questions = site.fetch('questions', 
                                  tagged=tag,
                                  sort='votes',
                                  order='desc',
                                  filter='withbody')
            
            if 'items' not in questions or len(questions['items']) == 0:
                print(f"  ⚠ No questions found for tag '{tag}'")
                
                # Try without hyphen if it exists
                if '-' in tag:
                    alt_tag = tag.replace('-', '')
                    print(f"  Trying alternative: '{alt_tag}'")
                    questions = site.fetch('questions', 
                                         tagged=alt_tag,
                                         sort='votes',
                                         order='desc',
                                         filter='withbody')
                
                if 'items' not in questions or len(questions['items']) == 0:
                    stats['questions_by_tag'][tag] = 0
                    continue
            
            # Process questions
            items = questions['items'][:num_per_tag]
            tag_questions = []
            
            print(f"  Found {len(questions['items'])} questions, processing {len(items)}...")
            
            # Collect answer IDs for batch fetching
            answer_ids_to_fetch = []
            
            for q in items:
                question_data = {
                    'question_id': q.get('question_id'),
                    'title': q.get('title'),
                    'body': q.get('body', ''),
                    'tags': q.get('tags', []),
                    'score': q.get('score', 0),
                    'view_count': q.get('view_count', 0),
                    'answer_count': q.get('answer_count', 0),
                    'is_answered': q.get('is_answered', False),
                    'accepted_answer_id': q.get('accepted_answer_id'),
                    'creation_date': q.get('creation_date'),
                    'link': q.get('link'),
                    'owner': q.get('owner', {}).get('display_name', 'unknown'),
                    'collected_tag': tag,
                    'accepted_answer': None
                }
                
                if question_data['accepted_answer_id']:
                    answer_ids_to_fetch.append(question_data['accepted_answer_id'])
                    stats['questions_with_accepted_answers'] += 1
                
                tag_questions.append(question_data)
            
            # Batch fetch answers (more efficient than one by one)
            if answer_ids_to_fetch:
                print(f"  Fetching {len(answer_ids_to_fetch)} accepted answers...")
                
                try:
                    # API allows up to 100 IDs at once
                    for i in range(0, len(answer_ids_to_fetch), 100):
                        batch_ids = answer_ids_to_fetch[i:i+100]
                        
                        answers = site.fetch('answers',
                                           ids=batch_ids,
                                           filter='withbody')
                        
                        if 'items' in answers:
                            # Create a mapping of answer_id to answer data
                            answer_map = {a['answer_id']: a for a in answers['items']}
                            
                            # Add answers to questions
                            for q in tag_questions:
                                if q['accepted_answer_id'] in answer_map:
                                    answer_data = answer_map[q['accepted_answer_id']]
                                    q['accepted_answer'] = {
                                        'body': answer_data.get('body', ''),
                                        'score': answer_data.get('score', 0),
                                        'is_accepted': answer_data.get('is_accepted', False),
                                        'creation_date': answer_data.get('creation_date')
                                    }
                                    stats['questions_with_answers_fetched'] += 1
                        
                        # Rate limiting between batches
                        if i + 100 < len(answer_ids_to_fetch):
                            time.sleep(1)
                            
                except Exception as e:
                    print(f"  ⚠ Error fetching answers: {e}")
                    stats['errors'].append(f"Answer fetch error for tag {tag}: {e}")
            
            # Update statistics
            stats['questions_by_tag'][tag] = len(tag_questions)
            stats['total_questions'] += len(tag_questions)
            stats['actual_tags_used'].append(tag)
            
            # Add to collection
            all_questions.extend(tag_questions)
            
            print(f"  ✓ Collected {len(tag_questions)} questions")
            print(f"  ✓ Fetched {len([q for q in tag_questions if q['accepted_answer']])} answers")
            print(f"  Total collected: {stats['total_questions']}")
            
            # Rate limiting
            if tag_index < len(tags):
                time.sleep(2)
                
        except Exception as e:
            error_msg = f"Error with tag '{tag}': {str(e)}"
            print(f"  ✗ {error_msg}")
            stats['errors'].append(error_msg)
            stats['questions_by_tag'][tag] = 0
    
    # Save data
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Main data file
    questions_file = os.path.join(output_dir, f'automotive_questions_with_answers_{timestamp}.json')
    with open(questions_file, 'w', encoding='utf-8') as f:
        json.dump(all_questions, f, indent=2, ensure_ascii=False)
    
    # Statistics file
    stats_file = os.path.join(output_dir, f'collection_stats_{timestamp}.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    # Create a summary CSV for quick review
    summary_file = os.path.join(output_dir, f'questions_summary_{timestamp}.csv')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("question_id,title,score,has_accepted_answer,tags\n")
        for q in all_questions:
            title = q['title'].replace(',', ';').replace('\n', ' ')
            tags = '|'.join(q['tags'])
            has_answer = 'Yes' if q['accepted_answer'] else 'No'
            f.write(f"{q['question_id']},{title},{q['score']},{has_answer},{tags}\n")
    
    # Print summary
    print(f"\n{'='*60}")
    print("COLLECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total questions collected: {stats['total_questions']}")
    print(f"Questions with accepted answers: {stats['questions_with_accepted_answers']}")
    print(f"Answers successfully fetched: {stats['questions_with_answers_fetched']}")
    print(f"\nQuestions by tag:")
    for tag, count in stats['questions_by_tag'].items():
        if count > 0:
            print(f"  {tag}: {count}")
    
    if stats['errors']:
        print(f"\n⚠ Errors: {len(stats['errors'])}")
    
    print(f"\nFiles saved:")
    print(f"  📄 {questions_file}")
    print(f"  📊 {stats_file}")
    print(f"  📋 {summary_file}")
    
    return all_questions, stats

def collect_reddit_questions(
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    user_agent: str = "CKG-RAG Data Collection v1.0",
    limit: int = 500,
    min_upvotes: int = 10
) -> tuple:
    """
    Collect automotive questions from Reddit r/MechanicAdvice
    
    Args:
        client_id: Reddit API client ID
        client_secret: Reddit API client secret
        user_agent: User agent for Reddit API
        limit: Maximum number of posts to collect
        min_upvotes: Minimum upvotes threshold
    
    Returns:
        Tuple of (questions_list, statistics_dict)
    """
    
    # Check for Reddit credentials
    if not client_id or not client_secret:
        # Try to load from environment or config file
        config_file = 'reddit_config.json'
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
                client_id = config.get('client_id')
                client_secret = config.get('client_secret')
        else:
            print("\n" + "="*60)
            print("REDDIT API CREDENTIALS REQUIRED")
            print("="*60)
            print("\nTo collect Reddit data, you need to:")
            print("1. Go to https://www.reddit.com/prefs/apps")
            print("2. Create a new app (select 'script' type)")
            print("3. Note your client_id and client_secret")
            print("\nThen create 'reddit_config.json' with:")
            print(json.dumps({"client_id": "your_id", "client_secret": "your_secret"}, indent=2))
            print("\n" + "="*60)
            return [], {"error": "No Reddit credentials provided"}
    
    print(f"\nCollecting Reddit r/MechanicAdvice posts...")
    print(f"Target: {limit} posts with {min_upvotes}+ upvotes")
    
    try:
        # Initialize Reddit instance
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        
        # Access the subreddit
        subreddit = reddit.subreddit("MechanicAdvice")
        
        # Storage for questions
        reddit_questions = []
        
        # Statistics
        stats = {
            'total_posts_checked': 0,
            'posts_collected': 0,
            'posts_with_solved_flair': 0,
            'posts_with_questions': 0,
            'posts_meeting_upvotes': 0,
            'collection_timestamp': datetime.now().isoformat()
        }
        
        # Question detection patterns
        question_patterns = [
            r'\?',  # Contains question mark
            r'^(why|what|how|when|where|is|are|can|could|should|would|will|do|does|did)',  # Starts with question word
        ]
        
        def is_question(text: str) -> bool:
            """Check if text contains a question"""
            text_lower = text.lower()
            for pattern in question_patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return True
            return False
        
        # Fetch posts from different time periods for variety
        time_filters = ['year', 'month', 'week']
        posts_per_filter = limit // len(time_filters)
        
        for time_filter in time_filters:
            print(f"\nFetching top posts from past {time_filter}...")
            
            try:
                # Get top posts
                posts = subreddit.top(time_filter=time_filter, limit=posts_per_filter * 2)
                
                for post in posts:
                    stats['total_posts_checked'] += 1
                    
                    # Check if we've collected enough
                    if stats['posts_collected'] >= limit:
                        break
                    
                    # Progress indicator
                    if stats['total_posts_checked'] % 50 == 0:
                        print(f"  Checked {stats['total_posts_checked']} posts, collected {stats['posts_collected']}")
                    
                    # Filter criteria
                    has_solved_flair = post.link_flair_text and 'solved' in post.link_flair_text.lower()
                    meets_upvotes = post.score >= min_upvotes
                    contains_question = is_question(post.title) or is_question(post.selftext)
                    
                    # Update statistics
                    if has_solved_flair:
                        stats['posts_with_solved_flair'] += 1
                    if meets_upvotes:
                        stats['posts_meeting_upvotes'] += 1
                    if contains_question:
                        stats['posts_with_questions'] += 1
                    
                    # Apply filters
                    if meets_upvotes and contains_question:
                        # Get top comment as potential answer
                        top_comment = None
                        try:
                            post.comments.replace_more(limit=0)
                            if post.comments:
                                top_comment = max(post.comments, key=lambda c: c.score if hasattr(c, 'score') else 0)
                        except:
                            pass
                        
                        # Extract post data
                        question_data = {
                            'source': 'reddit',
                            'subreddit': 'MechanicAdvice',
                            'post_id': post.id,
                            'title': post.title,
                            'body': post.selftext,
                            'url': f"https://reddit.com{post.permalink}",
                            'score': post.score,
                            'upvote_ratio': post.upvote_ratio,
                            'num_comments': post.num_comments,
                            'created_utc': post.created_utc,
                            'author': str(post.author) if post.author else 'deleted',
                            'flair': post.link_flair_text,
                            'is_solved': has_solved_flair,
                            'top_comment': {
                                'body': top_comment.body,
                                'score': top_comment.score,
                                'author': str(top_comment.author) if top_comment.author else 'deleted'
                            } if top_comment else None,
                            'time_filter': time_filter
                        }
                        
                        reddit_questions.append(question_data)
                        stats['posts_collected'] += 1
                        
            except Exception as e:
                print(f"  Error fetching {time_filter} posts: {e}")
                continue
        
        print(f"\n✓ Collected {stats['posts_collected']} Reddit posts")
        print(f"  Total checked: {stats['total_posts_checked']}")
        print(f"  With 'Solved' flair: {stats['posts_with_solved_flair']}")
        print(f"  Meeting upvote threshold: {stats['posts_meeting_upvotes']}")
        print(f"  Containing questions: {stats['posts_with_questions']}")
        
        return reddit_questions, stats
        
    except Exception as e:
        print(f"\n✗ Error connecting to Reddit: {e}")
        return [], {"error": str(e)}


def combine_datasets():
    """
    Combine Stack Exchange and Reddit datasets into unified format
    """
    print("\n" + "="*60)
    print("COMBINING DATASETS")
    print("="*60)
    
    combined_data = {
        'metadata': {
            'creation_date': datetime.now().isoformat(),
            'sources': ['stack_exchange', 'reddit'],
            'total_questions': 0
        },
        'questions': []
    }
    
    # Load Stack Exchange data
    se_files = [f for f in os.listdir('data') if 'automotive_questions_with_answers' in f and f.endswith('.json')]
    if se_files:
        latest_se = sorted(se_files)[-1]
        with open(os.path.join('data', latest_se), 'r') as f:
            se_data = json.load(f)
            print(f"Loaded {len(se_data)} Stack Exchange questions")
            
            # Convert to unified format
            for q in se_data:
                unified_q = {
                    'id': f"se_{q['question_id']}",
                    'source': 'stack_exchange',
                    'title': q['title'],
                    'question': q['body'],
                    'answer': q['accepted_answer']['body'] if q.get('accepted_answer') else None,
                    'score': q['score'],
                    'tags': q['tags'],
                    'url': q['link'],
                    'metadata': {
                        'view_count': q.get('view_count', 0),
                        'answer_count': q.get('answer_count', 0),
                        'author': q.get('owner', 'unknown')
                    }
                }
                combined_data['questions'].append(unified_q)
    
    # Load Reddit data
    reddit_files = [f for f in os.listdir('data') if 'reddit_questions' in f and f.endswith('.json')]
    if reddit_files:
        latest_reddit = sorted(reddit_files)[-1]
        with open(os.path.join('data', latest_reddit), 'r') as f:
            reddit_data = json.load(f)
            print(f"Loaded {len(reddit_data)} Reddit questions")
            
            # Convert to unified format
            for q in reddit_data:
                unified_q = {
                    'id': f"reddit_{q['post_id']}",
                    'source': 'reddit',
                    'title': q['title'],
                    'question': q['body'],
                    'answer': q['top_comment']['body'] if q.get('top_comment') else None,
                    'score': q['score'],
                    'tags': [q['flair']] if q.get('flair') else [],
                    'url': q['url'],
                    'metadata': {
                        'upvote_ratio': q.get('upvote_ratio', 0),
                        'num_comments': q.get('num_comments', 0),
                        'author': q.get('author', 'deleted'),
                        'is_solved': q.get('is_solved', False)
                    }
                }
                combined_data['questions'].append(unified_q)
    
    combined_data['metadata']['total_questions'] = len(combined_data['questions'])
    
    # Save combined dataset
    output_file = os.path.join('data', f'combined_automotive_qa_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Combined dataset saved: {output_file}")
    print(f"  Total questions: {combined_data['metadata']['total_questions']}")
    
    return combined_data


if __name__ == "__main__":
    print("CKG-RAG Paper - Automotive Questions Collection")
    print("="*60)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Collect automotive Q&A data')
    parser.add_argument('--source', choices=['stackexchange', 'reddit', 'both'], 
                       default='both', help='Data source to collect from')
    parser.add_argument('--se-limit', type=int, default=100, 
                       help='Questions per tag for Stack Exchange')
    parser.add_argument('--reddit-limit', type=int, default=500,
                       help='Number of Reddit posts to collect')
    
    args = parser.parse_args()
    
    data_collected = []
    
    # Collect Stack Exchange data
    if args.source in ['stackexchange', 'both']:
        print(f"\n[1/2] Collecting Stack Exchange data...")
        se_questions, se_stats = collect_automotive_questions_with_answers(num_per_tag=args.se_limit)
        if se_questions:
            data_collected.append('stack_exchange')
    
    # Collect Reddit data
    if args.source in ['reddit', 'both']:
        print(f"\n[2/2] Collecting Reddit data...")
        reddit_questions, reddit_stats = collect_reddit_questions(limit=args.reddit_limit)
        if reddit_questions and 'error' not in reddit_stats:
            # Save Reddit data
            reddit_file = os.path.join('data', f'reddit_questions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            with open(reddit_file, 'w', encoding='utf-8') as f:
                json.dump(reddit_questions, f, indent=2, ensure_ascii=False)
            print(f"Reddit data saved: {reddit_file}")
            data_collected.append('reddit')
    
    # Combine datasets if we have both
    if len(data_collected) > 1:
        combine_datasets()
    
    print("\n" + "="*60)
    print("DATA COLLECTION COMPLETED")
    print("="*60)