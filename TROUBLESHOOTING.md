# 🔧 Troubleshooting Guide

## Common Issues and Solutions

### 1. Services Won't Start

**Symptom:** `docker-compose up` fails or containers exit immediately

**Solutions:**

```bash
# Check Docker is running
docker ps

# Check logs
docker-compose logs

# Restart Docker daemon (Linux)
sudo systemctl restart docker

# Rebuild images
docker-compose build --no-cache
docker-compose up -d
```

### 2. Port Already in Use

**Symptom:** `Error: port is already allocated`

**Solution:**

```bash
# Check what's using the port
lsof -i :7474  # For Neo4j HTTP
lsof -i :7687  # For Neo4j Bolt
lsof -i :8888  # For Jupyter

# Kill the process or change ports in docker-compose.yml
ports:
  - "8889:8888"  # Use different host port
```

### 3. Neo4j Won't Start

**Symptom:** Neo4j container keeps restarting

**Checks:**

```bash
# View Neo4j logs
docker-compose logs neo4j

# Common issues:
# - Insufficient memory (needs 2GB+)
# - License issue (needs NEO4J_ACCEPT_LICENSE_AGREEMENT=yes)
# - Corrupted data (remove volumes and restart)
```

**Solutions:**

```bash
# Remove volumes and restart
docker-compose down -v
docker-compose up -d

# Increase Docker memory limit
# Docker Desktop: Settings > Resources > Memory > 8GB
```

### 4. Jupyter Connection Refused

**Symptom:** Cannot access http://localhost:8888

**Solutions:**

```bash
# Check if Jupyter is running
docker-compose ps jupyter

# View Jupyter logs
docker-compose logs jupyter

# Restart Jupyter
docker-compose restart jupyter

# Access from inside container
docker-compose exec jupyter jupyter notebook list
```

### 5. Neo4j Connection Failed in Python

**Symptom:** `ServiceUnavailable` or `AuthError` in notebook

**Solutions:**

```python
# Check connection parameters
from graph import GraphConnector

graph = GraphConnector(
    uri="bolt://neo4j:7687",  # Use 'neo4j' not 'localhost' inside Docker
    user="neo4j",
    password="graphml2024"
)

# Test connection
if graph.test_connection():
    print("Connected!")
```

**Alternative: Check from browser**
- Open http://localhost:7474
- Try connecting with credentials
- If that works, Python connection should work too

### 6. Data Loading Fails

**Symptom:** CSV loading returns errors or 0 rows

**Solutions:**

```bash
# Check CSV URLs are accessible
curl -I https://bit.ly/3JjgKgZ
curl -I https://bit.ly/3S9Uyd8

# Try alternative: Download locally first
wget https://bit.ly/3JjgKgZ -O /path/to/data/streams.csv

# Load from local file
LOAD CSV WITH HEADERS FROM 'file:///streams.csv' AS row
```

### 7. Node2Vec Fails

**Symptom:** `gds.node2vec.write` returns error

**Common Issues:**

```cypher
// 1. Graph projection doesn't exist
CALL gds.graph.list() YIELD graphName
// If empty, create projection first

// 2. Using deprecated procedure
// Use: gds.node2vec.write
// Not: gds.beta.node2vec.write

// 3. Insufficient memory
// Increase Neo4j heap:
NEO4J_dbms_memory_heap_max__size: 4G
```

### 8. Model Training Fails

**Symptom:** Out of memory or poor performance

**Solutions:**

```python
# Reduce data size
df_sample = df.sample(frac=0.5, random_state=42)

# Reduce model complexity
model = RandomForestClassifier(
    n_estimators=50,  # Reduce from 100
    max_depth=5,      # Limit depth
    n_jobs=2          # Limit parallelism
)

# Use smaller embeddings
ml.run_node2vec(embedding_dimension=4)  # Instead of 8
```

### 9. Jupyter Kernel Dies

**Symptom:** Kernel keeps dying or restarting

**Solutions:**

```bash
# Increase Docker memory
# Docker Desktop: Settings > Resources > Memory

# Restart Jupyter
docker-compose restart jupyter

# Clear outputs before running
# Jupyter: Cell > All Output > Clear

# Run cells individually instead of "Run All"
```

### 10. Slow Performance

**Symptom:** Queries or training taking too long

**Optimizations:**

```cypher
// Create indexes
CREATE INDEX stream_id FOR (s:Stream) ON (s.streamId);
CREATE INDEX stream_language FOR (s:Stream) ON (s.language);

// Use LIMIT in exploratory queries
MATCH (s:Stream)
RETURN s
LIMIT 100;

// Profile queries to find bottlenecks
PROFILE
MATCH (s:Stream)-[:SHARED_AUDIENCE]-(s2:Stream)
RETURN count(*);
```

```python
# Sample data for visualization
df_sample = df.sample(n=1000)

# Use smaller test_size
train_test_split(X, y, test_size=0.1)  # Instead of 0.2

# Reduce cross-validation folds
cross_val_score(model, X, y, cv=3)  # Instead of 5
```

## 🆘 Getting More Help

### Check Logs

```bash
# All logs
docker-compose logs

# Specific service
docker-compose logs neo4j
docker-compose logs jupyter

# Follow logs
docker-compose logs -f --tail=100
```

### Inspect Containers

```bash
# Container status
docker-compose ps

# Resource usage
docker stats

# Enter container
docker-compose exec jupyter /bin/bash
docker-compose exec neo4j /bin/bash
```

### Neo4j Diagnostics

```cypher
// Check database status
SHOW DATABASES;

// Check constraints
SHOW CONSTRAINTS;

// Check indexes
SHOW INDEXES;

// Check node counts
MATCH (n) RETURN labels(n), count(n);

// Check memory
CALL dbms.listConfig() 
YIELD name, value 
WHERE name CONTAINS 'memory';
```

### Reset Everything

If all else fails:

```bash
# Nuclear option: Remove everything and start fresh
docker-compose down -v
docker system prune -a
docker volume prune

# Rebuild from scratch
docker-compose build --no-cache
docker-compose up -d
```

## 📚 Additional Resources

- [Neo4j Documentation](https://neo4j.com/docs/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Jupyter Documentation](https://jupyter.org/documentation)
- [Neo4j Community Forum](https://community.neo4j.com/)

## 💬 Still Stuck?

1. Check container logs: `docker-compose logs`
2. Verify services are running: `docker-compose ps`
3. Test connections manually (Neo4j Browser, Jupyter)
4. Try the "Reset Everything" steps above
5. Check GitHub Issues (if this is from a repository)
