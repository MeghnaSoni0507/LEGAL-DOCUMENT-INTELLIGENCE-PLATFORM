# webapp/Dockerfile
# --- build stage ---
FROM node:18-alpine AS build
WORKDIR /app

# copy package metadata first to leverage cache
COPY package*.json ./

# install dependencies (use npm ci if you have package-lock.json)
RUN npm ci --silent

# copy source and build
COPY . .
ARG REACT_APP_API_BASE=""
ENV NODE_ENV=production
RUN npm run build

# --- production: nginx serves the static files ---
FROM nginx:stable-alpine AS prod

# ensure wget is available for healthcheck
RUN apk add --no-cache wget

# clear default nginx html
RUN rm -rf /usr/share/nginx/html/*

# copy built assets
COPY --from=build /app/build /usr/share/nginx/html

# copy optional custom nginx config (if you don't have one, container uses default nginx config)
# make sure this file is UTF-8 without BOM (we fixed this earlier)
COPY nginx.conf /etc/nginx/conf.d/default.conf

# healthcheck (uses wget from apk above)
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --quiet --tries=1 --spider http://localhost/ || exit 1

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
