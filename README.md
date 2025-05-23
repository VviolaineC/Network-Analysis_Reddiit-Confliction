## Mapping Online Conflict: A Study of Inter-Subreddit Dynamics on Reddit
![4f7a6dbd-ea1c-4f22-85de-a550afaa02b2](https://github.com/user-attachments/assets/ba8a82e0-5047-49c8-b564-8a00ec62726b)

### Basic Content:
1. Background
2. Research Method
3. Findings
4. Intervention Strategies
5. Conclusion


### Background:
Think: What happens when two Reddit communities go to war online? 
![ChatGPT Image 2025年5月5日 15_03_00](https://github.com/user-attachments/assets/babcab70-9b6c-49cd-8d89-e60d18913b0d)

### Research Method:
#### 📊 Data Source and Analysis Objects
The dataset is from the Stanford Network Analysis Project, covering the years 2005 - 2017
It contains over 330,000 cross - subreddit links, involving over 67,000 subreddits
Each link represents a post in one subreddit linking to a post in another subreddit
Each link in the data comes with an emotion label (positive/neutral or negative)


#### 🔍 Key Points of Analysis 
Focus on cross-community links with negative labels 
Negative links are commonly found in hostile behaviors such as mockery, criticism, and attacks 
Regard these links as direct signals of community conflicts

![656d7a13-1cfe-4484-9daf-03ef8900ed2a](https://github.com/user-attachments/assets/564b8271-25c2-4764-88bf-ca2238625294)

See: [process_reddit_data.py](https://github.com/VviolaineC/Network-Analysis_Reddiit-Confliction/blob/655e14ed1eb8f2c3fa9f50feb68600f2ed800c90/process_reddit_data.py)

### Findings:

#### 📊 Structural Characteristics of the Reddit Community Network
Interactions between communities are highly unbalanced (with obvious power - law distribution characteristics).

On average, each subreddit links to only about 5 other communities.
A few “supernodes” are highly active:

🔗 r/AskReddit: Receives links from 5,000+ communities (the largest inbound node).

🔗 r/SubredditDrama: Links to 3,000+ other communities (the largest outbound node).

Conclusion: A few “hub - type” communities dominate link propagation.



### ⚔️ The Three Roles in Reddit Conflict: Who Starts It, Who Gets Hit, Who Spreads It

**🧨 Conflict Instigators**

* Actively initiate a large number of negative cross-links
* Examples: r/ShitGnomeSays, r/imablue (over 90% of outbound links are hostile)
* Think of them as **aggressive niche communities**—small in size, but highly provocative

**🥺 Conflict Targets**

* Frequently attacked by others, rarely strike back
* Example: r/sjwnews receives many negative inbound links but sends few outbound links
* These are **the quiet victims** in Reddit’s inter-community conflicts

**🌉 Bridge Communities**

* Don’t start conflicts, but **amplify and spread them** through exposure
* Example: r/SubredditDrama serves as a **conflict hub**, connecting otherwise unrelated communities
* Think of them as **the gossip broadcasters**, making sure everyone hears about the drama

See: [network_analysis.py](https://github.com/VviolaineC/Network-Analysis_Reddiit-Confliction/blob/39cf7ab93acf7e9ebac2e92fa5e01b0e76d8ec80/network_analysis.py)


### 🛠️ Moderation Strategies to Reduce Inter-Subreddit Conflict

**1️⃣ Monitor Bridge Hubs**
Closely track bridge communities (e.g., r/SubredditDrama) that connect many subreddits

Throttle or flag hubs if they trigger a sudden surge of hostile cross-posts

Prevent local drama from turning into platform-wide conflict

**2️⃣ Rein in Aggressors**
Identify instigator subreddits with unusually high rates of outbound negativity

Use automated systems to flag toxic patterns

Reduce their visibility and reach, and prompt moderation action

**3️⃣ Protect Bystanders**
Adjust algorithms so controversial posts don’t flood large general-interest subs (e.g., r/AskReddit)

Encourage big communities to adopt “no external drama” rules

Create a buffer to stop toxic content from going viral



