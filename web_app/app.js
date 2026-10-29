/**
 * CEAP 3D Latent Explorer • High-Performance WebGL Engine
 * Visualizador Interativo de 1.34M Documentos através de 4 Espaços Latentes
 */

(function () {
  'use strict';

  // --- Elementos do DOM ---
  const container = document.getElementById('canvas-container');
  const loadingOverlay = document.getElementById('loading-overlay');
  const loadingText = document.getElementById('loading-text');
  const modelTabs = document.querySelectorAll('.tab-btn');
  const modelBadge = document.getElementById('model-badge');
  const modelTitle = document.getElementById('model-title');
  const modelDesc = document.getElementById('model-desc');
  const metricDim = document.getElementById('metric-dim');
  const metricInter = document.getElementById('metric-inter');
  const metricIntra = document.getElementById('metric-intra');
  const sliderSize = document.getElementById('slider-size');
  const sizeVal = document.getElementById('size-val');
  const sliderUnmapped = document.getElementById('slider-unmapped');
  const unmappedVal = document.getElementById('unmapped-val');
  const legendList = document.getElementById('legend-list');
  const classSearch = document.getElementById('class-search');
  const btnShowAll = document.getElementById('btn-show-all');
  const btnHideUnmapped = document.getElementById('btn-hide-unmapped');
  const btnResetCam = document.getElementById('btn-reset-cam');
  const btnToggleAutorotate = document.getElementById('btn-toggle-autorotate');
  const btnFullscreen = document.getElementById('btn-fullscreen');
  const btnTogglePanel = document.getElementById('btn-toggle-panel');
  const sidePanel = document.getElementById('side-panel');
  const pointCountBadge = document.getElementById('point-count-badge');
  const totalAnnotatedCount = document.getElementById('total-annotated-count');

  // --- Estado da Aplicação ---
  let metadata = null;
  let currentModelKey = 'metric';
  const coordsCache = {}; // Cache de Float32Array para cada modelo
  let classIds = null;    // Uint8Array com ID de classe (0..15)
  let nPoints = 0;
  let pointCloud = null;
  let pointsGeometry = null;
  let pointsMaterial = null;
  let isHideUnmapped = false;
  let isolatedClassId = null; // null = todas ativas; número = isolada

  // Trilha de visibilidade de classes para o Shader (16 classes)
  const classVisibility = new Float32Array(16).fill(1.0);

  // --- Three.js Setup ---
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x090d16);
  scene.fog = new THREE.FogExp2(0x090d16, 0.005);

  const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 1000);
  camera.position.set(0, 30, 95);

  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  container.appendChild(renderer.domElement);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.autoRotate = false;
  controls.autoRotateSpeed = 0.8;
  controls.target.set(0, 0, 0);

  // Grid sutil no chão
  const grid = new THREE.GridHelper(100, 20, 0x1e293b, 0x111827);
  grid.position.y = -35;
  scene.add(grid);

  // Luzes sutis
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
  scene.add(ambientLight);

  // --- Utilitários de Interface ---
  function showLoading(msg) {
    loadingText.textContent = msg;
    loadingOverlay.classList.remove('hidden');
  }

  function hideLoading() {
    loadingOverlay.classList.add('hidden');
  }

  function formatNumber(num) {
    return new Intl.NumberFormat('pt-BR').format(num);
  }

  // --- Descompressão Pako Segura com Alinhamento de Bytes ---
  function decompressGzFloat32(arrayBuffer) {
    const inflated = pako.inflate(new Uint8Array(arrayBuffer));
    if (inflated.byteOffset % 4 === 0) {
      return new Float32Array(inflated.buffer, inflated.byteOffset, inflated.byteLength / 4);
    }
    const alignedBuffer = new Uint8Array(inflated.byteLength);
    alignedBuffer.set(inflated);
    return new Float32Array(alignedBuffer.buffer, 0, alignedBuffer.byteLength / 4);
  }

  function decompressGzUint8(arrayBuffer) {
    return pako.inflate(new Uint8Array(arrayBuffer));
  }

  // --- Carregamento de Dados ---
  async function loadMetadata() {
    const res = await fetch('data/metadata.json');
    if (!res.ok) throw new Error('Falha ao carregar metadata.json');
    return await res.json();
  }

  async function loadClasses() {
    const res = await fetch('data/classes.bin.gz');
    if (!res.ok) throw new Error('Falha ao carregar classes.bin.gz');
    const buf = await res.arrayBuffer();
    return decompressGzUint8(buf);
  }

  async function loadModelCoords(modelKey) {
    if (coordsCache[modelKey]) {
      return coordsCache[modelKey];
    }
    const modelInfo = metadata.models[modelKey];
    showLoading(`Carregando coordenadas do modelo ${modelInfo.name}...`);
    const res = await fetch(`data/${modelInfo.file}`);
    if (!res.ok) throw new Error(`Falha ao carregar ${modelInfo.file}`);
    const buf = await res.arrayBuffer();
    showLoading('Descompactando geometria 3D...');
    const floatArray = decompressGzFloat32(buf);
    coordsCache[modelKey] = floatArray;
    return floatArray;
  }

  // --- Shader Customizado de Alta Performance ---
  const vertexShader = `
    attribute float aClassId;
    attribute vec3 color;
    varying vec3 vColor;
    varying float vAlpha;
    varying float vDiscard;

    uniform float uClassVisible[16];
    uniform float uGlobalSize;
    uniform float uUnmappedAlpha;

    void main() {
      int cid = int(clamp(aClassId + 0.5, 0.0, 15.0));
      float vis = uClassVisible[cid];
      
      if (vis < 0.05) {
        vDiscard = 1.0;
        gl_PointSize = 0.0;
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
      }
      
      vDiscard = 0.0;
      vColor = color;
      
      if (cid == 0) {
        vAlpha = uUnmappedAlpha;
      } else {
        vAlpha = 1.0;
      }
      
      if (vAlpha < 0.005) {
        vDiscard = 1.0;
        gl_PointSize = 0.0;
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
      }
      
      vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
      float classMultiplier = (cid == 0) ? 0.85 : 1.75;
      gl_PointSize = uGlobalSize * classMultiplier * (120.0 / -mvPosition.z);
      gl_Position = projectionMatrix * mvPosition;
    }
  `;

  const fragmentShader = `
    varying vec3 vColor;
    varying float vAlpha;
    varying float vDiscard;

    void main() {
      if (vDiscard > 0.5 || vAlpha <= 0.005) discard;
      vec2 coord = gl_PointCoord - vec2(0.5);
      float dist = length(coord);
      if (dist > 0.5) discard;
      
      // Suavização sutil da borda circular
      float edgeAlpha = smoothstep(0.5, 0.35, dist);
      gl_FragColor = vec4(vColor, vAlpha * edgeAlpha);
    }
  `;

  // --- Inicialização da Nuvem de Pontos ---
  function initPointCloud(initialCoords) {
    pointsGeometry = new THREE.BufferGeometry();

    // 1. Posições (dinâmicas conforme o modelo)
    pointsGeometry.setAttribute('position', new THREE.BufferAttribute(initialCoords, 3));

    // 2. Cores e IDs de Classe (fixos para os 1.34M docs)
    const colorArray = new Float32Array(nPoints * 3);
    const classIdArray = new Float32Array(nPoints);

    for (let i = 0; i < nPoints; i++) {
      const cid = classIds[i];
      classIdArray[i] = cid;
      const pal = metadata.palette[cid];
      if (pal && pal.rgb) {
        colorArray[i * 3 + 0] = pal.rgb[0];
        colorArray[i * 3 + 1] = pal.rgb[1];
        colorArray[i * 3 + 2] = pal.rgb[2];
      } else {
        colorArray[i * 3 + 0] = 0.32;
        colorArray[i * 3 + 1] = 0.32;
        colorArray[i * 3 + 2] = 0.35;
      }
    }

    pointsGeometry.setAttribute('color', new THREE.BufferAttribute(colorArray, 3));
    pointsGeometry.setAttribute('aClassId', new THREE.BufferAttribute(classIdArray, 1));

    // 3. Material
    pointsMaterial = new THREE.ShaderMaterial({
      vertexShader: vertexShader,
      fragmentShader: fragmentShader,
      uniforms: {
        uClassVisible: { value: classVisibility },
        uGlobalSize: { value: parseFloat(sliderSize.value) },
        uUnmappedAlpha: { value: parseFloat(sliderUnmapped.value) / 100.0 }
      },
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending
    });

    pointCloud = new THREE.Points(pointsGeometry, pointsMaterial);
    scene.add(pointCloud);
  }

  // --- Troca Instantânea de Modelo ---
  async function switchModel(modelKey) {
    if (modelKey === currentModelKey && pointCloud) return;

    currentModelKey = modelKey;
    const modelInfo = metadata.models[modelKey];

    // Atualizar UI das Abas
    modelTabs.forEach((tab) => {
      tab.classList.toggle('active', tab.getAttribute('data-model') === modelKey);
    });

    // Atualizar Painel Lateral
    modelBadge.textContent = modelInfo.badge;
    modelBadge.style.color = modelInfo.color;
    modelBadge.style.borderColor = `${modelInfo.color}55`;
    modelBadge.style.backgroundColor = `${modelInfo.color}1a`;

    modelTitle.textContent = modelInfo.name;
    modelDesc.textContent = modelInfo.description;
    metricDim.textContent = `${modelInfo.dims}-D`;
    metricInter.textContent = modelInfo.inter_dist;
    metricIntra.textContent = modelInfo.intra_sim;

    try {
      const coords = await loadModelCoords(modelKey);
      if (!pointCloud) {
        initPointCloud(coords);
      } else {
        pointsGeometry.attributes.position.array.set(coords);
        pointsGeometry.attributes.position.needsUpdate = true;
      }
      hideLoading();
    } catch (err) {
      console.error(err);
      alert('Erro ao carregar modelo: ' + err.message);
      hideLoading();
    }
  }

  // --- Construção da Legenda Interativa ---
  function buildLegend() {
    legendList.innerHTML = '';

    // Ordenar classes por contagem (Não Mapeado no início ou no fim)
    const sortedCids = Object.keys(metadata.palette).sort((a, b) => {
      if (a === '0') return 1;
      if (b === '0') return -1;
      return metadata.palette[b].count - metadata.palette[a].count;
    });

    sortedCids.forEach((cidStr) => {
      const cid = parseInt(cidStr, 10);
      const item = metadata.palette[cidStr];

      const div = document.createElement('div');
      div.className = 'legend-item';
      div.dataset.cid = cid;

      div.innerHTML = `
        <div class="legend-item-left">
          <span class="class-dot" style="background:${item.hex}; box-shadow: 0 0 6px ${item.hex}88;"></span>
          <span class="class-name">${item.name}</span>
        </div>
        <span class="class-count">${formatNumber(item.count)}</span>
      `;

      // Clique para isolar ou alternar classe
      div.addEventListener('click', () => {
        toggleIsolateClass(cid);
      });

      legendList.appendChild(div);
    });
  }

  // --- Isolamento / Filtro de Classes ---
  function toggleIsolateClass(cid) {
    if (isolatedClassId === cid) {
      // Já estava isolado -> restaurar todas
      showAllClasses();
    } else {
      isolatedClassId = cid;
      for (let i = 0; i < 16; i++) {
        classVisibility[i] = (i === cid) ? 1.0 : 0.0;
      }
      updateLegendVisuals();
    }
  }

  function showAllClasses() {
    isolatedClassId = null;
    for (let i = 0; i < 16; i++) {
      classVisibility[i] = (i === 0 && isHideUnmapped) ? 0.0 : 1.0;
    }
    updateLegendVisuals();
  }

  function updateLegendVisuals() {
    const items = legendList.querySelectorAll('.legend-item');
    items.forEach((item) => {
      const cid = parseInt(item.dataset.cid, 10);
      const isVis = classVisibility[cid] > 0.5;
      item.classList.toggle('dimmed', !isVis);
    });
  }

  // --- Event Listeners dos Controles ---
  function setupEventListeners() {
    // Abas de Modelo
    modelTabs.forEach((btn) => {
      btn.addEventListener('click', () => {
        const model = btn.getAttribute('data-model');
        switchModel(model);
      });
    });

    // Slider de Tamanho dos Pontos
    sliderSize.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      sizeVal.textContent = `${val.toFixed(1)}px`;
      if (pointsMaterial) {
        pointsMaterial.uniforms.uGlobalSize.value = val;
      }
    });

    // Slider de Opacidade dos Não Mapeados
    sliderUnmapped.addEventListener('input', (e) => {
      const val = parseInt(e.target.value, 10);
      unmappedVal.textContent = `${val}%`;
      if (pointsMaterial) {
        pointsMaterial.uniforms.uUnmappedAlpha.value = val / 100.0;
      }
    });

    // Botão Mostrar Todas
    btnShowAll.addEventListener('click', () => {
      isHideUnmapped = false;
      btnHideUnmapped.textContent = 'Ocultar Não Mapeados';
      showAllClasses();
    });

    // Botão Ocultar / Mostrar Não Mapeados
    btnHideUnmapped.addEventListener('click', () => {
      isHideUnmapped = !isHideUnmapped;
      classVisibility[0] = isHideUnmapped ? 0.0 : 1.0;
      btnHideUnmapped.textContent = isHideUnmapped ? 'Mostrar Não Mapeados' : 'Ocultar Não Mapeados';
      updateLegendVisuals();
    });

    // Busca de Classes na Legenda
    classSearch.addEventListener('input', (e) => {
      const q = e.target.value.trim().toLowerCase();
      const items = legendList.querySelectorAll('.legend-item');
      items.forEach((item) => {
        const name = item.querySelector('.class-name').textContent.toLowerCase();
        if (name.includes(q)) {
          item.style.display = 'flex';
        } else {
          item.style.display = 'none';
        }
      });
    });

    // Reset de Câmera
    btnResetCam.addEventListener('click', () => {
      camera.position.set(0, 30, 95);
      controls.target.set(0, 0, 0);
    });

    // Alternar Rotação Automática
    btnToggleAutorotate.addEventListener('click', () => {
      controls.autoRotate = !controls.autoRotate;
      btnToggleAutorotate.classList.toggle('active', controls.autoRotate);
    });

    // Tela Cheia
    btnFullscreen.addEventListener('click', () => {
      if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().catch(() => {});
      } else {
        document.exitFullscreen().catch(() => {});
      }
    });

    // Ocultar / Mostrar Painel Lateral
    btnTogglePanel.addEventListener('click', () => {
      sidePanel.classList.toggle('collapsed');
      btnTogglePanel.classList.toggle('active', !sidePanel.classList.contains('collapsed'));
    });

    // Redimensionamento da Janela
    window.addEventListener('resize', () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    });
  }

  // --- Loop de Renderização (60 FPS) ---
  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }

  // --- Inicialização Geral ---
  async function main() {
    showLoading('Carregando metadados do CEAP...');
    try {
      metadata = await loadMetadata();
      nPoints = metadata.n_points;

      pointCountBadge.textContent = `${(nPoints / 1000000).toFixed(2)}M docs`;
      totalAnnotatedCount.textContent = `${formatNumber(metadata.total_annotated)} docs`;

      buildLegend();
      setupEventListeners();

      showLoading('Carregando classes dos 1.34M documentos...');
      classIds = await loadClasses();

      // Carregar modelo padrão (Deep Metric Learning ArcFace)
      await switchModel(metadata.default_model || 'metric');

      animate();
    } catch (err) {
      console.error('Falha na inicialização:', err);
      alert('Erro ao carregar dados da visualização: ' + err.message);
      hideLoading();
    }
  }

  // Start!
  main();
})();
