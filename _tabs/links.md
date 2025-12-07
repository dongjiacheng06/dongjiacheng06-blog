---
title: 友链
layout: page
icon: fas fa-link
order: 5
---

欢迎互换友链，如需添加请通过邮箱或留言联系我。

<style>
.friend-links {
  display: grid;
  grid-template-columns: 1fr;
  gap: 1rem;
  max-width: 720px;
  margin: 1rem auto;
}

.friend-card {
  display: flex;
  gap: 0.75rem;
  padding: 1rem;
  border: 1px solid var(--friend-border, #e1e4e8);
  border-radius: 12px;
  background: var(--friend-bg, rgba(0, 0, 0, 0.02));
  box-shadow: 0 6px 18px rgba(0, 0, 0, 0.04);
}

.friend-card img {
  width: 64px;
  height: 64px;
  border-radius: 50%;
  object-fit: cover;
  border: 1px solid var(--friend-border, #e1e4e8);
}

.friend-meta {
  display: flex;
  flex-direction: column;
  gap: 0.15rem;
}

.friend-meta h2 {
  margin: 0;
  font-size: 1.05rem;
}

.friend-meta p {
  margin: 0;
}

.friend-meta .subtitle {
  color: var(--text-muted-color, #6c757d);
  font-weight: 600;
}

.friend-meta .description {
  color: var(--text-color, inherit);
}
</style>

{% if site.data.friends %}
<div class="friend-links">
  {% for friend in site.data.friends %}
  <div class="friend-card">
    <img src="{{ friend.avatar }}" alt="{{ friend.name }} avatar" loading="lazy">
    <div class="friend-meta">
      <h2><a href="{{ friend.url }}" target="_blank" rel="noopener noreferrer">{{ friend.name }}</a></h2>
      {% if friend.subtitle %}<p class="subtitle">{{ friend.subtitle }}</p>{% endif %}
      {% if friend.description %}<p class="description">{{ friend.description }}</p>{% endif %}
    </div>
  </div>
  {% endfor %}
</div>
{% else %}
<p>友链整理中，欢迎联系添加。</p>
{% endif %}
