---
layout: post
author: Colin Swaney
title: The Linked List Cycle Problem
date: 2019-09-17
categories: [programming]
category: programming
tags: [programming]
excerpt: "<p></p>"
---

I've been studying basic computer science concepts recently because I don't have a "traditional" programming background, but people like to ask these types of questions at interviews for some reason (don't ask me why). In any case, some of the practice problems are kind of fun, like the one I'm about to describe about linked lists. First, what is a linked list? It's simply a collection of objects in which each object holds a value and a pointer to one of the other objects. Linked lists are useful for implementing stacks and queues because it is easy to remove objects from the beginning or end of a linked list. (On the other hand, they aren't too useful as iterable objects due to the way the memory of the nodes is laid out). However, a linked list could contain a cycle in which the "last" object (usually called a "node") points back to an object that came before it. The question is simply how to detect whether a linked list contains such a cycle assuming that you've been told which object is the "first" object in the list.

There is an obvious answer to this question: iterate through the linked list, keeping track of each node you see in a dictionary, and return `true` if you find a node that points back to a node in your dictionary of seen nodes. This solution is $$\mathcal{O}(n)$$ in time and space. It turns out that there is a simple solution that reduces to the space complexity to $$\mathcal{O}(1)$$, which I'll now explain.

Let's imagine two people "walking" through the linked list. The first person walks slowly, moving one step each second; the second person walks twice as fast (two steps per second). We start our walkers out at the same time and stop each second we stop to see where each walker is. Now the claim is that the walkers will intersect at some point in time if and only if the linked list contains a cycle. Intuitively, if the walkers enter a cycle, we know that the faster walker will eventually catch up with the slower walker, and otherwise—if there is not cycle—the faster walker will always stay ahead of the slower walker, so certainly this method will return `true` if and only the linked list contains a cycle. But how can we be sure that the faster walker will catch the slower walker in $$\mathcal{O}(n)$$ time?

The reason boils down to the following fact: a person starting $$k$$ steps behind but walking twice as fast as another person will catch the slower person in precisely $$k$$ steps.

$$y = x$$ 

$$y = 2x - k$$

$$\rightarrow x = k$$

Now consider the second when our slow walker has just arrived at the beginning of a cycle, and suppose that the cycle contains $$k$$ nodes. Wherever our fast walker is, we can be sure that it is at most $$k$$ nodes ahead of the slower walker. In terms of catching up to the slow walker, that means the fast walker is at most $$k$$ nodes *behind*. But that means that she will *exactly* catch the slow walker in $$k$$ steps. So not only will our walkers cross, but they will in fact land on the same node after some number of steps, and that will always happen within at worst $$n$$ steps. 