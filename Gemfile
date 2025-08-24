# frozen_string_literal: true

source "https://rubygems.org"

# Jekyll和主题
gem "jekyll", "~> 4.3"
# gem "jekyll-theme-chirpy", "~> 7.0", ">= 7.0.1"  # 注释掉，使用remote_theme

# 必需的插件  
gem "jekyll-paginate"
gem "jekyll-sitemap" 
gem "jekyll-gist"
gem "jekyll-feed"
gem "jekyll-include-cache"
gem "jekyll-archives"
gem "jekyll-remote-theme"  # 支持remote_theme
gem "jekyll-seo-tag"  # Chirpy主题必需
gem "jekyll-redirect-from"  # Chirpy主题必需

# 测试
gem "html-proofer", "~> 5.0", group: :test

# Windows和JRuby支持
platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

gem "wdm", "~> 0.1.1", :platforms => [:mingw, :x64_mingw, :mswin]

# JRuby特定
gem "http_parser.rb", "~> 0.6.0", :platforms => [:jruby]