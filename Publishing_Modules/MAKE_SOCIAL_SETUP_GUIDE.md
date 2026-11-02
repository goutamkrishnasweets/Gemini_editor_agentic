# Make.com Multi-Platform Social Publishing Guide (Instagram & Facebook Reels)

This guide provides the battle-tested, step-by-step instructions to configure **Make.com** for automatic, 1-click video publishing across **Instagram Reels** and **Facebook Pages** from AMTCE.

---

## 📌 Overview

```
[AMTCE Bot / Telegram] ──(Webhook POST)──> [ Make.com Scenario ]
                                                ├──> [ Instagram for Business: Create a Reel Post ]
                                                └──> [ Facebook Pages: Upload Video from URL ]
```

Every time you approve a video in Telegram, AMTCE automatically sends the video URL, title, hashtags, and caption to Make.com, which publishes to all connected social channels simultaneously.

---

## 🛠️ Step 1: Create the Webhook in Make.com

1. Go to [Make.com](https://us2.make.com/) (or your Make region) and log in.
2. In the left sidebar, click **Webhooks** (plug icon).
3. In the top-right corner, click **Add**.
4. Name your webhook: `Gemini-Editor` (or any preferred name).
5. Leave IP restrictions blank and click **Save**.
6. Click **Copy address to clipboard**. The link looks like:
   `https://hook.us2.make.com/j2pdpbgrvnv2crgxxmkd34vvslg9vsa0`
   *(⚠️ Note: Do NOT open this URL in your browser. It is a background data receiver for Python).*

---

## 🔗 Step 2: Save Webhook URL in Telegram

In your Telegram bot chat, send:
```text
/setmakewebhookurl <PASTED_WEBHOOK_URL>
```
*(Example: `/setmakewebhookurl https://hook.us2.make.com/j2pdpbgrvnv2crgxxmkd34vvslg9vsa0`)*

The bot will reply: `✅ Make.com Webhook URL Synced!`.

---

## 🎬 Step 3: Build the Scenario in Make.com

### 3.1 Create Scenario & Attach Webhook
1. On the left sidebar menu of Make.com, click **Scenarios** (connecting dots icon).
2. Click **Create a new scenario** (top-right purple button).
3. Click the big **`+`** circle in the center.
4. Search for **`Webhooks`** and select **Custom webhook**.
5. In the **Webhook \*** dropdown, select your webhook: **`Gemini-Editor`**.
6. Click **Save / OK**.

---

### 3.2 Add Instagram for Business (Reels)
1. On the right side of the red Webhook circle, click the half-circle with the **`+`** icon.
2. Search for **`Instagram for Business`** and click it.
3. Select **`Create a reel post`** *(⚠️ Make sure you select **Reel post**, NOT "Photo post")*.
4. In the settings box:
   - **Connection \***: Click **Add** to log into Facebook and connect your Instagram account.
   - **Page \***: Select your Instagram Page (e.g. `Fitsbydisha (@fitsbydisha)`).  
     *(⚠️ Keep the `Map` switch next to Page **OFF**).*
   - **Video URL \***: Type or select:
     ```text
     {{1.video_url}}
     ```
   - **Cover Frame**: Leave blank (default is 0).
   - **Caption**: Type or select:
     ```text
     {{1.caption}}
     ```
5. Click the purple **Save** button.

---

### 3.3 Add Facebook Pages (Video / Reels)
1. On the right side of the pink **Instagram** circle, click the half-circle with the **`+`** icon.
2. Search for **`Facebook Pages`** and click it.
3. Select **`Upload a Video`**.
4. In the settings box:
   - **Connection \***: Select your existing Facebook connection.
   - **Page \***: Select your Facebook Page from the dropdown (e.g. `Fitsbydisha (Delhi)`).  
     *(⚠️ Keep the `Map` switch next to Page **OFF**).*
   - **Type \***: Click the dropdown and select **`download a video from URL`**.  
     *(⚠️ Do NOT turn on the `Map` switch next to Type. Make sure the dropdown says `download a video from URL`).*
   - **URL \***: A clean box will appear. Enter:
     ```text
     {{1.video_url}}
     ```
   - **Video caption**: Enter:
     ```text
     {{1.caption}}
     ```
5. Click the purple **Save** button.

---

## 🚀 Step 4: Save & Activate

1. At the bottom toolbar of Make.com, click the **Save** button (next to *Run once*).
2. In the top-right corner, flip the schedule switch from **`Inactive`** to **`Active`** (green).
3. Ensure scheduling is set to: **`Immediately as data arrives`**.

---

## 🛡️ Critical Pitfalls to Avoid

| Pitfall | Why It Breaks | What To Do Instead |
|---|---|---|
| Opening the webhook URL in Chrome | Chrome sends a `GET` request, returning `There is no scenario listening` | Never open webhook in browser; Python sends `POST` data in the background. |
| Clicking the dotted connector line / wrench between circles | Opens the **"Set up a filter"** modal (e.g. Label: AMTCE); if condition is empty or unmet, it blocks all videos | Do NOT configure filters on the connector lines. Click **Cancel** if opened. |
| Choosing Instagram "Create a photo post" | Instagram rejects video files on photo endpoints (`Photo URL * required`) | Always choose **`Create a reel post`** for videos. |
| Leaving Facebook Type as "upload a video" | Asks for raw file bytes (`File name *` and `Data *`) with red errors | Change the Type dropdown to **`download a video from URL`**. |
| Turning ON the `Map` switch next to Page or Type | Turns clean dropdown menus into raw numeric ID inputs | Keep the `Map` toggle **OFF** on dropdowns. |

---

## 🔄 Failover Guarantee
If Make.com is ever temporarily down, out of operations, or hits a rate limit, AMTCE automatically detects it and publishes directly via local direct uploaders (`Publishing_Modules/meta_uploader.py`). Nothing is lost.
