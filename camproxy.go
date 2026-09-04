// camproxy - authenticated DVR snapshot proxy for the Even Realities G2 glasses app.
//
// Runs on the dual-homed Linux server. Fetches JPEG snapshots from the camera
// subnet using HTTP digest auth, optionally converts to greyscale and downscales,
// caches briefly, and serves them over the VPN-reachable interface with CORS.
//
//	go mod init camproxy
//	go get github.com/icholy/digest golang.org/x/image/draw
//	CGO_ENABLED=0 go build -o camproxy .
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/jpeg"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/icholy/digest"
	xdraw "golang.org/x/image/draw"
)

type Camera struct {
	Name  string `json:"name"`  // URL slug, e.g. "front"
	Label string `json:"label"` // display name for the phone UI
	URL   string `json:"url"`   // http://192.168.100.15/cgi-bin/snapshot.cgi?channel=1
	User  string `json:"user"`
	Pass  string `json:"pass"`
}

type Config struct {
	Listen   string   `json:"listen"`    // ":9000"
	Token    string   `json:"token"`     // optional shared secret; empty = no auth
	CacheTTL string   `json:"cache_ttl"` // "3s"
	Timeout  string   `json:"timeout"`   // "8s"
	Quality  int      `json:"quality"`   // JPEG quality out, e.g. 70
	Cameras  []Camera `json:"cameras"`
}

type entry struct {
	data []byte
	at   time.Time
}

type server struct {
	cfg      Config
	ttl      time.Duration
	cams     map[string]Camera
	clients  map[string]*http.Client
	mu       sync.Mutex
	cache    map[string]entry
	inflight map[string]*sync.Mutex // collapse concurrent fetches per camera
}

func main() {
	path := flag.String("config", "config.json", "path to config file")
	flag.Parse()

	raw, err := os.ReadFile(*path)
	if err != nil {
		log.Fatalf("read config: %v", err)
	}
	var cfg Config
	if err := json.Unmarshal(raw, &cfg); err != nil {
		log.Fatalf("parse config: %v", err)
	}
	if cfg.Listen == "" {
		cfg.Listen = ":9000"
	}
	if cfg.Quality == 0 {
		cfg.Quality = 70
	}
	ttl, err := time.ParseDuration(orDefault(cfg.CacheTTL, "3s"))
	if err != nil {
		log.Fatalf("cache_ttl: %v", err)
	}
	timeout, err := time.ParseDuration(orDefault(cfg.Timeout, "8s"))
	if err != nil {
		log.Fatalf("timeout: %v", err)
	}

	s := &server{
		cfg:      cfg,
		ttl:      ttl,
		cams:     map[string]Camera{},
		clients:  map[string]*http.Client{},
		cache:    map[string]entry{},
		inflight: map[string]*sync.Mutex{},
	}
	for _, c := range cfg.Cameras {
		s.cams[c.Name] = c
		s.clients[c.Name] = &http.Client{
			Timeout:   timeout,
			Transport: &digest.Transport{Username: c.User, Password: c.Pass},
		}
		s.inflight[c.Name] = &sync.Mutex{}
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /cams", s.handleList)
	mux.HandleFunc("GET /cam/{name}", s.handleSnapshot)
	mux.HandleFunc("OPTIONS /", func(w http.ResponseWriter, r *http.Request) {
		cors(w)
		w.WriteHeader(http.StatusNoContent)
	})
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, "ok\n")
	})

	log.Printf("camproxy listening on %s with %d cameras", cfg.Listen, len(s.cams))
	srv := &http.Server{
		Addr:              cfg.Listen,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}
	log.Fatal(srv.ListenAndServe())
}

// handleList returns the camera roster so the phone can self-configure.
func (s *server) handleList(w http.ResponseWriter, r *http.Request) {
	cors(w)
	if !s.authorized(r) {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return
	}
	type item struct {
		Name  string `json:"name"`
		Label string `json:"label"`
	}
	out := make([]item, 0, len(s.cfg.Cameras))
	for _, c := range s.cfg.Cameras {
		out = append(out, item{Name: c.Name, Label: c.Label})
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(out)
}

// handleSnapshot serves /cam/{name}?w=640&gray=1
func (s *server) handleSnapshot(w http.ResponseWriter, r *http.Request) {
	cors(w)
	if !s.authorized(r) {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return
	}
	name := r.PathValue("name")
	cam, ok := s.cams[name]
	if !ok {
		http.Error(w, "unknown camera", http.StatusNotFound)
		return
	}

	width := atoiDefault(r.URL.Query().Get("w"), 640)
	gray := r.URL.Query().Get("gray") != "0"
	key := fmt.Sprintf("%s|%d|%t", name, width, gray)

	// Collapse concurrent requests for the same camera so the DVR sees one fetch.
	lock := s.inflight[name]
	lock.Lock()
	defer lock.Unlock()

	if b, ok := s.get(key); ok {
		writeJPEG(w, b)
		return
	}

	raw, err := s.fetch(cam)
	if err != nil {
		log.Printf("fetch %s: %v", name, err)
		http.Error(w, "upstream error", http.StatusBadGateway)
		return
	}
	out, err := transform(raw, width, gray, s.cfg.Quality)
	if err != nil {
		log.Printf("transform %s: %v", name, err)
		http.Error(w, "decode error", http.StatusBadGateway)
		return
	}
	s.put(key, out)
	writeJPEG(w, out)
}

func (s *server) fetch(cam Camera) ([]byte, error) {
	resp, err := s.clients[cam.Name].Get(cam.URL)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("dvr returned %s", resp.Status)
	}
	return io.ReadAll(io.LimitReader(resp.Body, 16<<20))
}

// transform decodes, optionally converts to greyscale, scales to width, re-encodes.
func transform(raw []byte, width int, gray bool, quality int) ([]byte, error) {
	src, err := jpeg.Decode(bytes.NewReader(raw))
	if err != nil {
		return nil, err
	}
	b := src.Bounds()
	if width <= 0 || width > b.Dx() {
		width = b.Dx()
	}
	height := b.Dy() * width / b.Dx()
	dstRect := image.Rect(0, 0, width, height)

	var dst image.Image
	if gray {
		g := image.NewGray(dstRect)
		xdraw.CatmullRom.Scale(g, dstRect, src, b, xdraw.Src, nil)
		dst = g
	} else {
		c := image.NewRGBA(dstRect)
		xdraw.CatmullRom.Scale(c, dstRect, src, b, xdraw.Src, nil)
		dst = c
	}

	buf := &bytes.Buffer{}
	if err := jpeg.Encode(buf, dst, &jpeg.Options{Quality: quality}); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func (s *server) get(key string) ([]byte, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	e, ok := s.cache[key]
	if !ok || time.Since(e.at) > s.ttl {
		return nil, false
	}
	return e.data, true
}

func (s *server) put(key string, data []byte) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cache[key] = entry{data: data, at: time.Now()}
}

func (s *server) authorized(r *http.Request) bool {
	if s.cfg.Token == "" {
		return true
	}
	if r.URL.Query().Get("token") == s.cfg.Token {
		return true
	}
	return r.Header.Get("X-Auth-Token") == s.cfg.Token
}

func cors(w http.ResponseWriter) {
	h := w.Header()
	h.Set("Access-Control-Allow-Origin", "*")
	h.Set("Access-Control-Allow-Headers", "X-Auth-Token")
	h.Set("Access-Control-Allow-Methods", "GET, OPTIONS")
}

func writeJPEG(w http.ResponseWriter, b []byte) {
	w.Header().Set("Content-Type", "image/jpeg")
	w.Header().Set("Content-Length", strconv.Itoa(len(b)))
	w.Header().Set("Cache-Control", "no-store")
	w.Write(b)
}

func orDefault(v, d string) string {
	if v == "" {
		return d
	}
	return v
}

func atoiDefault(v string, d int) int {
	n, err := strconv.Atoi(v)
	if err != nil || n <= 0 {
		return d
	}
	return n
}
