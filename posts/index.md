---
layout: post-list
title: All Posts
excerpt: "A List of Posts"
comments: false
---
	<h1>{{ page.title }}</h1>
	<ul class="posts">

	  {% for post in site.posts %}
	    <li><span>{{ post.date | date_to_string }}</span> » <a href="{{ post.url }}" title="{{ post.title }}">{{ post.title }}</a></li>
	  {% endfor %}
	</ul>