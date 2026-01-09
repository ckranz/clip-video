# SEO Technical Specifications

Quick reference for exact SEO specifications, thresholds, and metrics.

---

## On-Page Elements

### Title Tags

| Specification | Value |
|---------------|-------|
| Minimum length | 50 characters |
| Maximum length | 60 characters |
| Maximum pixel width | 600 pixels |

**Rules:**
- Front-load primary keyword at the beginning
- Use power words and numbers to increase CTR
- Match search intent precisely

### Meta Descriptions

| Specification | Value |
|---------------|-------|
| Desktop maximum | 160 characters |
| Mobile maximum | 120 characters |
| Recommended minimum | 150 characters |

**Template:** `[Value proposition] + [Key benefit] + [CTA]`

### URL Structure

| Specification | Value |
|---------------|-------|
| Maximum length | 75 characters |
| Optimal word count | 5-7 semantic words |

**Rules:**
- Use hyphens to separate words
- Remove dates from evergreen content URLs
- Avoid special characters, parameters, session IDs

### Headers

- One H1 per page containing primary keyword
- Never skip hierarchy levels (H1→H2→H3, not H1→H3)
- H1 should match or relate to title tag
- Question-based headers improve AEO

### Internal Linking

| Specification | Value |
|---------------|-------|
| Maximum click depth | 3 clicks from homepage |
| Links per page | 3-10 |
| Audit frequency | Monthly |

**Architecture:** Hub-Spoke-Bridge model
- **Hubs:** Pillar pages (link out to spokes, receive links from spokes)
- **Spokes:** Deep dives and FAQs (link back to hub, 1-2 sibling links)
- **Bridges:** Comparison pages connecting clusters

---

## Core Web Vitals

### Largest Contentful Paint (LCP)

| Rating | Threshold |
|--------|-----------|
| Good | ≤ 2.5 seconds |
| Needs Improvement | 2.5 - 4.0 seconds |
| Poor | > 4.0 seconds |

### Interaction to Next Paint (INP)

| Rating | Threshold |
|--------|-----------|
| Good | ≤ 200 milliseconds |
| Needs Improvement | 200 - 500 milliseconds |
| Poor | > 500 milliseconds |

### Cumulative Layout Shift (CLS)

| Rating | Threshold |
|--------|-----------|
| Good | ≤ 0.1 |
| Needs Improvement | 0.1 - 0.25 |
| Poor | > 0.25 |

**Requirement:** 75% of page visits must meet "Good" threshold (75th percentile)

---

## Images

### File Size Limits

| Image Type | Maximum Size |
|------------|--------------|
| Hero images | 200 KB |
| Content images | 150 KB |
| Thumbnails | 50 KB |

### Format Recommendations

- **WebP:** 25-35% better compression than JPEG (primary format)
- **AVIF:** 50% better compression (progressive enhancement)

### Alt Text

- Maximum length: 125 characters
- Front-load the keyword
- Describe what's IN the image
- Avoid "image of" prefixes
- Use empty `alt=""` for decorative images

### Lazy Loading

- **Do:** Lazy load below-fold images
- **Don't:** Never lazy load above-fold images (hurts LCP)

---

## Mobile

| Specification | Value |
|---------------|-------|
| Touch target minimum | 48×48 pixels |
| Minimum font size | 16px |

**Requirements:**
- Responsive design across all devices
- No horizontal scrolling
- Forms easy to complete on mobile
- Core Web Vitals pass on mobile

---

## Content

| Specification | Value |
|---------------|-------|
| Recommended minimum words | 2,000 |
| AI-only content quality score | 3.6/10 |
| AI-assisted with human quality score | 7.5/10 |
| Content audit frequency | Quarterly |

---

## Schema Markup

Priority schema types: Article, FAQPage, HowTo, Product, Review, LocalBusiness, Organization, BreadcrumbList, Person

Page 1 sites using schema: 72.6%

---

## Implementation Checklist

### New Websites
1. Mobile-responsive design
2. HTTPS security
3. Core Web Vitals optimization
4. Title tags and meta descriptions
5. Basic schema markup (Organization, WebPage)
6. Content with proper header hierarchy
7. Image optimization (WebP, alt text, dimensions)
8. Internal linking structure
9. Google Business Profile (if local)
10. Basic E-E-A-T signals (author bios, about page)

### Existing Websites
1. Technical SEO audit
2. Content audit and refresh strategy
3. Schema markup implementation
4. Internal linking optimization
5. E-E-A-T enhancement

---

## Reference Sources

- [Google Search Central](https://developers.google.com/search/docs)
- [Web.dev Core Web Vitals](https://web.dev/explore/learn-core-web-vitals)
- [Schema.org](https://schema.org/)
- [Google Rich Results Test](https://search.google.com/test/rich-results)
- [PageSpeed Insights](https://pagespeed.web.dev/)
