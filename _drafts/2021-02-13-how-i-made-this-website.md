---
layout: post
author: Colin Swaney
title: How I Made This Website
date: 2021-06-2
categories: [development]
category: development
tags: [development]
excerpt: "<p>A template for creating a modern-ish blog with Jekyll.</p>"
---

Outline
I. Introduction
- Something about choices when making an academic/research blog
- Overview of Jekyll
  - What use Jekyll?
  - How does it work?

II. Step-by-Step
- Jekyll
  - Can jump right in to how to setup/config b/c already introduced
  - Configuration (leave install etc. to Jekyll website, but mention rbenv!)
    - Syntax highlighting
    - LaTeX
- UIKit
  - Brief intro
  - Directions...
- Styling?


There are plenty of options out there for technical folks looking to create a blog. Many of them provide nice features and styling out-of-the-box (perhaps at a price). I've seen nice Wordpress examples and have heard good things about Ghost. Something along these lines is probably the way to go for someone that is just getting started.. But for those of us that insist on DIY, the solution of choice is Jekyll.

After you learn a little bit about front-end web development and develop some general programming chops, you'll realize that creating a full-blown blog isn't really a scary proposition. The frontend tends to be pretty damn simple and there are many frameworks to help. Most of the work boils down to record keeping and converting text files into HTML.

- There are two basic approaches to generating the blog.
  1. Generate all HTML pages on the backend. With this approach, everything (well, most things) get done in advance and served up "complete". Every article has a corresponding HTML file and pages get populated with categories, article counts, and so forth based on the articles found. Basically, you write template HTML files for pages of the website and write plain text files for the articles and then run an engine that takes everything and creates the site. This is the approach taken by Jekyll.
  2. With modern front-end tooling, there is no real reason (besides speed, perhaps) that everything needs to be "compiled" in advance. In effect, the blog is a presentation of a database, which happens to consist of a text files. Personally, I think this makes life a lot simpler and elegant. 


Jekyll is static site *generator*: it takes text files (and, optionally, HTML) and creates a static website. Advantages of Jekyll are:
1. Many people use Jekyll. You will typically find answers to questions (and you're going to have questions!) and there are a ton of plug-ins and themes available.
2. Jekyll works with GitHub pages. This means that you can host your blog for free, and updating your blog is as simple as pushing to remote.
<!-- 356 words -->


<!-- Jekyll -->
Follow the instructions on the Jekyll website to install Jekyll and its dependencies. I also recommended that you install `rbenv` to manage Ruby environments (this will make it easier to get a hold of and use a version of Ruby that in compatible with Jekyll, which may not work perfectly with the lastest Ruby release).

The other things to do are:

- Create a new Jekyll project using the default settings
- Custom the project by adding `_includes`, `_layouts`, and `index.html`.
- Add support for LaTeX.

To add support for LaTeX, first modify the `_config.yml` file to use `kramdown` with `mathjax`:

```yml
# Build settings
markdown: kramdown
kramdown:
  math_engine: mathjax
theme: minima
plugins:
  - jekyll-feed
```

Next, in the projects `default.html`, include `KaTeX` in the `head`:

```html
<script src="//code.jquery.com/jquery-1.11.1.min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.js"></script>
```


<!-- UIKit -->
I really like UIKit. Not only did I use UIKit to create the layout for the website, but I also cloned quite a bit of the styling from the UIKit website. To include UIKit for layout, all you need to do is download the core JavaScript and CSS modules from the UIKit website and throw them in your source files. Specifically, you'll want to add:

```bash
/assets/css/uikit.css
/assets/css/uikit.min.css
/assets/js/uikit.js
/assets/js/uikit.min.js
/assets/js/uikit-icons.js
/assets/js/uikit-icons.min.js
```

In order to clone the UIKit website styling, first get a copy of the CSS:

```bash
cd ./assets/css
wget ... theme.css
```

Then, you'll want to get a copy of the fonts used by the UIKit website:

```
mkdir ./assets/fonts
cd ./assets/fonts
wget ... montserrat-600.woff
...
```

<!-- Syntax Highlighting -->
The Jekyll setup includes an HTML syntax highliger, `Rouge`, which tags `pre` components with the `highlighter-rouge` class, and adds additional tags to text within `pre` components. In order to actually get syntax highlighting, you'll need to add some additional styling. You can create your own CSS, but `Rouge` includes a commmand-line tool to generate themed CSS. For example, to create a CSS file with the default `Rougify` theme, run

```bash
rougify style base16 > /assets/css/syntax.css
```