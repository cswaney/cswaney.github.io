---
layout: post
author: Colin Swaney
title: How I Made This Website
date: 2021-02-13
categories: [development]
category: development
tags: [development]
excerpt: "<p></p>"
---

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