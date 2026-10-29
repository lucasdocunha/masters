/**
 * CEAP 3D Latent Explorer & Estúdio de Rotulação Ativa (Label Studio Mode)
 * - Visualizador WebGL de 1.34M Documentos (Three.js Shaders)
 * - Estúdio de Rotulação Dedicado com Google Docs Viewer Iframe
 * - Criação Dinâmica de Classes com Paleta Hexadecimal
 * - Filas Inteligentes por Proximidade de Cluster
 * - Sincronização em Tempo Real com Tabela, CSV e GPU 3D
 */

(function () {
  'use strict';

  // --- Elementos do DOM: Navegação Global & 3D ---
  const container = document.getElementById('canvas-container');
  const loadingOverlay = document.getElementById('loading-overlay');
  const loadingText = document.getElementById('loading-text');
  const modelTabs = document.getElementById('model-tabs');
  const modelTabBtns = document.querySelectorAll('.tab-btn');
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
  const btnExportCSV = document.getElementById('btn-export-csv');

  // --- Elementos do DOM: Seletor de Telas ---
  const view3D = document.getElementById('view-3d');
  const viewStudio = document.getElementById('view-studio');
  const viewTable = document.getElementById('view-table');
  const btnView3D = document.getElementById('btn-view-3d');
  const btnViewStudio = document.getElementById('btn-view-studio');
  const btnViewTable = document.getElementById('btn-view-table');
  const studioPendingBadge = document.getElementById('studio-pending-badge');

  // --- Elementos do DOM: Estúdio de Rotulação ---
  const studioQueueSelect = document.getElementById('studio-queue-select');
  const studioCounterText = document.getElementById('studio-counter-text');
  const studioPrevBtn = document.getElementById('studio-prev-btn');
  const studioNextBtn = document.getElementById('studio-next-btn');
  const studioSkipBtn = document.getElementById('studio-skip-btn');
  const studioAutoAdvance = document.getElementById('studio-auto-advance');
  const docIdPill = document.getElementById('doc-id-pill');
  const docCurrentBadge = document.getElementById('doc-current-badge');
  const docClusterProximity = document.getElementById('doc-cluster-proximity');
  const docSupplierVal = document.getElementById('doc-supplier-val');
  const docSubcotaVal = document.getElementById('doc-subcota-val');
  const docYearVal = document.getElementById('doc-year-val');
  const docCamaraLink = document.getElementById('doc-camara-link');
  const docReloadViewer = document.getElementById('doc-reload-viewer');
  const docFocus3D = document.getElementById('doc-focus-3d');
  const docIframe = document.getElementById('doc-iframe');
  const iframeLoader = document.getElementById('iframe-loader');
  const classPillsContainer = document.getElementById('class-pills-container');
  const btnToggleAddClass = document.getElementById('btn-toggle-add-class');
  const newClassForm = document.getElementById('new-class-form');
  const newClassName = document.getElementById('new-class-name');
  const newClassColor = document.getElementById('new-class-color');
  const btnConfirmAddClass = document.getElementById('btn-confirm-add-class');
  const btnCancelAddClass = document.getElementById('btn-cancel-add-class');
  const sessionCountText = document.getElementById('session-count-text');
  const btnStudioExportCSV = document.getElementById('btn-studio-export-csv');

  // --- Elementos do DOM: Tabela & CSV ---
  const tableSearchInput = document.getElementById('table-search-input');
  const tableFilterClass = document.getElementById('table-filter-class');
  const tableSortSelect = document.getElementById('table-sort-select');
  const tableTotalCount = document.getElementById('table-total-count');
  const btnTableDownloadCSV = document.getElementById('btn-table-download-csv');
  const btnTableDownloadJSON = document.getElementById('btn-table-download-json');
  const tableBody = document.getElementById('table-body');
  const pageInfoText = document.getElementById('page-info-text');
  const pageFirstBtn = document.getElementById('page-first-btn');
  const pagePrevBtn = document.getElementById('page-prev-btn');
  const pageCurrBtn = document.getElementById('page-curr-btn');
  const pageNextBtn = document.getElementById('page-next-btn');
  const pageLastBtn = document.getElementById('page-last-btn');

  // --- Elementos do DOM: Inspeção Estilo Plotly (Hover & Pinned) ---
  const btnToggleHover = document.getElementById('btn-toggle-hover');
  const hudPlotlyBadge = document.getElementById('hud-plotly-badge');
  const plotlyHoverTooltip = document.getElementById('plotly-hover-tooltip');
  const hoverClassDot = document.getElementById('hover-class-dot');
  const hoverClassName = document.getElementById('hover-class-name');
  const hoverDocId = document.getElementById('hover-doc-id');
  const hoverSupplier = document.getElementById('hover-supplier');
  const hoverSubcota = document.getElementById('hover-subcota');
  const hoverProximity = document.getElementById('hover-proximity');
  const hoverCoords = document.getElementById('hover-coords');

  const plotlyPinnedCard = document.getElementById('plotly-pinned-card');
  const pinnedClassDot = document.getElementById('pinned-class-dot');
  const pinnedDocTitle = document.getElementById('pinned-doc-title');
  const pinnedClassBadge = document.getElementById('pinned-class-badge');
  const btnClosePinned = document.getElementById('btn-close-pinned');
  const pinnedSupplier = document.getElementById('pinned-supplier');
  const pinnedSubcota = document.getElementById('pinned-subcota');
  const pinnedYear = document.getElementById('pinned-year');
  const pinnedCluster = document.getElementById('pinned-cluster');
  const pinnedCoords = document.getElementById('pinned-coords');
  const pinnedCamaraLink = document.getElementById('pinned-camara-link');
  const pinnedGoogleLink = document.getElementById('pinned-google-link');
  const btnPinnedOpenStudio = document.getElementById('btn-pinned-open-studio');

  // --- Estado da Aplicação ---
  let metadata = null;
  let currentModelKey = 'metric';
  const coordsCache = {};
  let classIds = null;
  let nPoints = 0;
  let pointCloud = null;
  let pointsGeometry = null;
  let pointsMaterial = null;
  let isHideUnmapped = false;
  let isolatedClassId = null;

  // Shader: Visibilidade de até 32 classes
  const classVisibility = new Float32Array(32).fill(1.0);

  // Dataset de Rotulação & Persistência
  let labelingDataset = [];
  let userAnnotations = {};
  let customClasses = {};
  let sessionLabeledCount = 0;

  // Estado da Inspeção Estilo Plotly
  let isHoverEnabled = true;
  let hoveredCoordsIdx = null;
  let pinnedCoordsIdx = null;
  let hoverReticle = null;
  let pinnedReticle = null;
  let isOrbitDragging = false;
  let lastHoverTimestamp = 0;
  let docIds = null;
  let urlPrefixes = null;
  let urlIndices = null;
  const coordsIdxToDoc = new Map();

  // Estado do Estúdio de Rotulação
  let studioQueue = [];
  let studioIndex = 0;
  let currentStudioDoc = null;

  // Estado da Tabela
  let tableFiltered = [];
  let tablePage = 1;
  const tablePageSize = 50;

  // Atalhos de teclado atribuídos dinamicamente
  const KEY_SHORTCUTS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'];
  const shortcutToClass = {};

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

  const grid = new THREE.GridHelper(100, 20, 0x1e293b, 0x111827);
  grid.position.y = -35;
  scene.add(grid);

  const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
  scene.add(ambientLight);

  // --- Utilitários ---
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

  function decompressGzJSON(arrayBuffer) {
    const inflated = pako.inflate(new Uint8Array(arrayBuffer));
    const str = new TextDecoder('utf-8').decode(inflated);
    return JSON.parse(str);
  }

  function decompressGzInt32(arrayBuffer) {
    const inflated = pako.inflate(new Uint8Array(arrayBuffer));
    if (inflated.byteOffset % 4 === 0) {
      return new Int32Array(inflated.buffer, inflated.byteOffset, inflated.byteLength / 4);
    }
    const alignedBuffer = new Uint8Array(inflated.byteLength);
    alignedBuffer.set(inflated);
    return new Int32Array(alignedBuffer.buffer, 0, alignedBuffer.byteLength / 4);
  }

  function decompressGzUint16(arrayBuffer) {
    const inflated = pako.inflate(new Uint8Array(arrayBuffer));
    if (inflated.byteOffset % 2 === 0) {
      return new Uint16Array(inflated.buffer, inflated.byteOffset, inflated.byteLength / 2);
    }
    const alignedBuffer = new Uint8Array(inflated.byteLength);
    alignedBuffer.set(inflated);
    return new Uint16Array(alignedBuffer.buffer, 0, alignedBuffer.byteLength / 2);
  }

  // --- Shader Customizado (Suporta até 32 classes dinâmicas) ---
  const vertexShader = `
    attribute float aClassId;
    attribute vec3 color;
    varying vec3 vColor;
    varying float vAlpha;
    varying float vDiscard;

    uniform float uClassVisible[32];
    uniform float uGlobalSize;
    uniform float uUnmappedAlpha;

    void main() {
      int cid = int(clamp(aClassId + 0.5, 0.0, 31.0));
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
      float edgeAlpha = smoothstep(0.5, 0.35, dist);
      gl_FragColor = vec4(vColor, vAlpha * edgeAlpha);
    }
  `;

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

  async function loadLabelingDataset() {
    try {
      const res = await fetch('data/labeling_dataset.json.gz');
      if (!res.ok) return [];
      const buf = await res.arrayBuffer();
      return decompressGzJSON(buf);
    } catch (e) {
      console.warn('Dataset de rotulação não encontrado:', e);
      return [];
    }
  }

  async function loadDocIds() {
    try {
      const res = await fetch('data/doc_ids.bin.gz');
      if (!res.ok) return null;
      const buf = await res.arrayBuffer();
      return decompressGzInt32(buf);
    } catch (e) {
      console.warn('doc_ids.bin.gz não disponível:', e);
      return null;
    }
  }

  async function loadUrlMapping() {
    try {
      const [resPref, resIdx] = await Promise.all([
        fetch('data/url_prefixes.json'),
        fetch('data/url_indices.bin.gz')
      ]);
      if (!resPref.ok || !resIdx.ok) return;
      urlPrefixes = await resPref.json();
      const bufIdx = await resIdx.arrayBuffer();
      urlIndices = decompressGzUint16(bufIdx);
    } catch (e) {
      console.warn('Mapeamento de URLs não disponível:', e);
    }
  }

  // --- Inicialização da Nuvem de Pontos ---
  function initPointCloud(initialCoords) {
    pointsGeometry = new THREE.BufferGeometry();
    pointsGeometry.setAttribute('position', new THREE.BufferAttribute(initialCoords, 3));

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

  // --- Troca Instantânea de Modelo 3D ---
  async function switchModel(modelKey) {
    if (modelKey === currentModelKey && pointCloud) return;

    currentModelKey = modelKey;
    const modelInfo = metadata.models[modelKey];

    modelTabBtns.forEach((tab) => {
      tab.classList.toggle('active', tab.getAttribute('data-model') === modelKey);
    });

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

      // Atualizar posição do marcador fixado se houver ponto selecionado
      if (pinnedCoordsIdx !== null && pinnedReticle && pointsGeometry) {
        const pos = pointsGeometry.attributes.position.array;
        const sx = pos[pinnedCoordsIdx * 3 + 0];
        const sy = pos[pinnedCoordsIdx * 3 + 1];
        const sz = pos[pinnedCoordsIdx * 3 + 2];
        pinnedReticle.position.set(sx, sy, sz);
        if (pinnedCoords) pinnedCoords.textContent = `X: ${sx.toFixed(2)} | Y: ${sy.toFixed(2)} | Z: ${sz.toFixed(2)}`;
      }

      hideLoading();
    } catch (err) {
      console.error(err);
      alert('Erro ao carregar modelo: ' + err.message);
      hideLoading();
    }
  }

  // --- Construção da Legenda 3D ---
  function buildLegend() {
    legendList.innerHTML = '';
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

      div.addEventListener('click', () => {
        toggleIsolateClass(cid);
      });

      legendList.appendChild(div);
    });
  }

  function toggleIsolateClass(cid) {
    if (isolatedClassId === cid) {
      showAllClasses();
    } else {
      isolatedClassId = cid;
      for (let i = 0; i < 32; i++) {
        classVisibility[i] = (i === cid) ? 1.0 : 0.0;
      }
      updateLegendVisuals();
    }
  }

  function showAllClasses() {
    isolatedClassId = null;
    for (let i = 0; i < 32; i++) {
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

  // =========================================================================
  // NAVEGAÇÃO ENTRE TELAS (3D vs ESTÚDIO vs TABELA)
  // =========================================================================
  function switchView(viewName) {
    [view3D, viewStudio, viewTable].forEach(v => v.classList.remove('active'));
    [btnView3D, btnViewStudio, btnViewTable].forEach(b => b.classList.remove('active'));

    if (viewName === '3d') {
      view3D.classList.add('active');
      btnView3D.classList.add('active');
      modelTabs.style.display = 'flex';
      btnTogglePanel.style.display = 'flex';
      btnResetCam.style.display = 'flex';
      btnToggleAutorotate.style.display = 'flex';
    } else if (viewName === 'studio') {
      viewStudio.classList.add('active');
      btnViewStudio.classList.add('active');
      modelTabs.style.display = 'none';
      btnTogglePanel.style.display = 'none';
      btnResetCam.style.display = 'none';
      btnToggleAutorotate.style.display = 'none';
      updateStudioView();
    } else if (viewName === 'table') {
      viewTable.classList.add('active');
      btnViewTable.classList.add('active');
      modelTabs.style.display = 'none';
      btnTogglePanel.style.display = 'none';
      btnResetCam.style.display = 'none';
      btnToggleAutorotate.style.display = 'none';
      applyTableFilters();
    }
  }

  // =========================================================================
  // MOTOR DO ESTÚDIO DE ROTULAÇÃO (LABEL STUDIO MODE)
  // =========================================================================
  function getClassIdByName(name) {
    for (const [cidStr, item] of Object.entries(metadata.palette)) {
      if (item.name.toLowerCase() === name.toLowerCase()) {
        return parseInt(cidStr, 10);
      }
    }
    return 0;
  }

  function getClassPaletteByName(name) {
    for (const item of Object.values(metadata.palette)) {
      if (item.name.toLowerCase() === name.toLowerCase()) {
        return item;
      }
    }
    return { name: name, hex: '#52525b', rgb: [0.32, 0.32, 0.35] };
  }

  function filterStudioQueue() {
    const qType = studioQueueSelect.value;
    if (qType === 'all_candidates') {
      studioQueue = labelingDataset.filter(d => d.class === 'Não Mapeado');
    } else if (qType === 'all_labeled') {
      studioQueue = labelingDataset.filter(d => d.class !== 'Não Mapeado');
    } else {
      // Filtrar pelo cluster mais próximo
      studioQueue = labelingDataset.filter(d => d.closest_cluster === qType && d.class === 'Não Mapeado');
      if (studioQueue.length === 0) {
        // Se já rotulou todos daquele cluster, mostrar os rotulados dele
        studioQueue = labelingDataset.filter(d => d.closest_cluster === qType);
      }
    }

    studioIndex = 0;
    updateStudioView();
  }

  function updateStudioView() {
    if (!studioQueue || studioQueue.length === 0) {
      docIdPill.textContent = 'Fila Vazia';
      docCurrentBadge.textContent = '-';
      docSupplierVal.textContent = 'Nenhum documento encontrado para este filtro.';
      docSubcotaVal.textContent = '-';
      docYearVal.textContent = '-';
      docClusterProximity.textContent = '-';
      docIframe.src = 'about:blank';
      iframeLoader.style.display = 'none';
      studioCounterText.textContent = '0 de 0';
      return;
    }

    if (studioIndex < 0) studioIndex = 0;
    if (studioIndex >= studioQueue.length) studioIndex = studioQueue.length - 1;

    currentStudioDoc = studioQueue[studioIndex];
    studioCounterText.textContent = `Doc ${studioIndex + 1} de ${studioQueue.length}`;

    // Preencher metadados
    docIdPill.textContent = `Doc #${currentStudioDoc.id}`;
    const pal = getClassPaletteByName(currentStudioDoc.class);
    docCurrentBadge.textContent = currentStudioDoc.class;
    docCurrentBadge.style.color = pal.hex;
    docCurrentBadge.style.borderColor = pal.hex;
    docCurrentBadge.style.backgroundColor = `${pal.hex}18`;

    const proxSim = (currentStudioDoc.similarity * 100).toFixed(1);
    docClusterProximity.textContent = `Mais próximo: ${currentStudioDoc.closest_cluster} (${proxSim}%)`;

    docSupplierVal.textContent = currentStudioDoc.fornecedor || 'Não Informado';
    docSubcotaVal.textContent = currentStudioDoc.subcota || 'Não Informado';
    docYearVal.textContent = currentStudioDoc.ano || '-';

    docCamaraLink.href = currentStudioDoc.url;

    // Carregar no iframe via Google Docs Viewer
    iframeLoader.style.display = 'flex';
    const googleViewerUrl = `https://docs.google.com/viewer?url=${encodeURIComponent(currentStudioDoc.url)}&embedded=true`;
    docIframe.src = googleViewerUrl;

    docIframe.onload = () => {
      iframeLoader.style.display = 'none';
    };

    // Atualizar pílulas de classe destacando a classe atual
    renderClassPills();
  }

  function renderClassPills() {
    classPillsContainer.innerHTML = '';
    const currentClass = currentStudioDoc ? currentStudioDoc.class : null;

    // Obter todas as classes conhecidas exceto 'Não Mapeado'
    const sortedClasses = Object.values(metadata.palette)
      .filter(p => p.name !== 'Não Mapeado')
      .sort((a, b) => b.count - a.count);

    let keyIdx = 0;
    sortedClasses.forEach((p) => {
      const btn = document.createElement('div');
      btn.className = 'class-pill-btn';
      if (currentClass === p.name) btn.classList.add('active');

      const shortcutKey = keyIdx < KEY_SHORTCUTS.length ? KEY_SHORTCUTS[keyIdx] : '';
      if (shortcutKey) {
        shortcutToClass[shortcutKey] = p.name;
      }
      keyIdx++;

      btn.innerHTML = `
        <div class="class-pill-left">
          <span class="class-pill-dot" style="background:${p.hex}; box-shadow: 0 0 6px ${p.hex};"></span>
          <span class="class-pill-name">${p.name}</span>
        </div>
        <div class="class-pill-right">
          ${shortcutKey ? `<span class="class-shortcut-key">${shortcutKey}</span>` : ''}
          <span class="class-pill-count">${formatNumber(p.count)}</span>
        </div>
      `;

      btn.addEventListener('click', () => {
        assignLabel(p.name);
      });

      classPillsContainer.appendChild(btn);
    });
  }

  // --- Atribuição de Rótulo (1-Click & Sincronização 3D/CSV) ---
  function assignLabel(newClassName) {
    if (!currentStudioDoc) return;

    const oldClass = currentStudioDoc.class;
    if (oldClass === newClassName) return;

    currentStudioDoc.class = newClassName;
    currentStudioDoc.status = 'rotulado';

    // 1. Gravar em localStorage
    userAnnotations[currentStudioDoc.id] = newClassName;
    localStorage.setItem('ceap_user_annotations', JSON.stringify(userAnnotations));

    // 2. Atualizar contadores
    sessionLabeledCount++;
    sessionCountText.textContent = `Nesta sessão: ${sessionLabeledCount} rotulados`;

    const oldCid = getClassIdByName(oldClass);
    const newCid = getClassIdByName(newClassName);

    if (metadata.palette[oldCid] && metadata.palette[oldCid].count > 0) {
      metadata.palette[oldCid].count--;
    }
    if (metadata.palette[newCid]) {
      metadata.palette[newCid].count++;
    }

    // 3. Atualizar GPU Three.js em tempo real
    if (pointsGeometry && currentStudioDoc.coords_idx !== undefined) {
      const idx = currentStudioDoc.coords_idx;
      if (classIds && idx < classIds.length) {
        classIds[idx] = newCid;
        pointsGeometry.attributes.aClassId.array[idx] = newCid;
        pointsGeometry.attributes.aClassId.needsUpdate = true;

        const newPal = metadata.palette[newCid];
        if (newPal && newPal.rgb) {
          pointsGeometry.attributes.color.array[idx * 3 + 0] = newPal.rgb[0];
          pointsGeometry.attributes.color.array[idx * 3 + 1] = newPal.rgb[1];
          pointsGeometry.attributes.color.array[idx * 3 + 2] = newPal.rgb[2];
          pointsGeometry.attributes.color.needsUpdate = true;
        }
      }
    }

    // 4. Atualizar UI
    buildLegend();
    updatePendingBadge();

    if (studioAutoAdvance.checked) {
      nextDoc();
    } else {
      updateStudioView();
    }
  }

  function nextDoc() {
    if (studioIndex < studioQueue.length - 1) {
      studioIndex++;
      updateStudioView();
    } else {
      updateStudioView();
    }
  }

  function prevDoc() {
    if (studioIndex > 0) {
      studioIndex--;
      updateStudioView();
    }
  }

  function updatePendingBadge() {
    const unmappedCount = labelingDataset.filter(d => d.class === 'Não Mapeado').length;
    studioPendingBadge.textContent = `${(unmappedCount / 1000).toFixed(1)}k`;
  }

  // --- Criação Dinâmica de Nova Classe ---
  function addNewCustomClass(name, hexColor) {
    const cleanName = name.trim().toLowerCase().replace(/\s+/g, '_');
    if (!cleanName) return;

    if (getClassIdByName(cleanName) !== 0) {
      alert(`A classe "${cleanName}" já existe!`);
      return;
    }

    const nextCid = Object.keys(metadata.palette).length;
    const r = parseInt(hexColor.slice(1, 3), 16) / 255.0;
    const g = parseInt(hexColor.slice(3, 5), 16) / 255.0;
    const b = parseInt(hexColor.slice(5, 7), 16) / 255.0;

    const newPalItem = {
      name: cleanName,
      hex: hexColor,
      rgb: [round(r, 4), round(g, 4), round(b, 4)],
      count: 0
    };

    metadata.palette[nextCid] = newPalItem;
    customClasses[cleanName] = newPalItem;
    localStorage.setItem('ceap_custom_classes', JSON.stringify(customClasses));

    // Reconstruir controles e aplicar ao doc atual
    buildLegend();
    renderClassPills();
    populateTableClassFilter();
    assignLabel(cleanName);

    newClassForm.classList.add('hidden');
    newClassName.value = '';
  }

  function round(num, decimals) {
    return Math.round(num * Math.pow(10, decimals)) / Math.pow(10, decimals);
  }

  // --- Focar Câmera 3D no Documento ---
  function focusDocIn3D(coordsIdx) {
    if (!pointCloud || coordsIdx === undefined) return;
    const pos = pointsGeometry.attributes.position.array;
    const x = pos[coordsIdx * 3 + 0];
    const y = pos[coordsIdx * 3 + 1];
    const z = pos[coordsIdx * 3 + 2];

    switchView('3d');
    controls.target.set(x, y, z);
    camera.position.set(x, y + 12, z + 28);
    pinPlotlyDoc(coordsIdx);
  }

  // =========================================================================
  // MOTOR DE INSPEÇÃO 3D ESTILO PLOTLY (HOVER TOOLTIP & PINNED CARD)
  // =========================================================================
  function createPlotlyReticles() {
    if (hoverReticle && pinnedReticle) return;

    // 1. Retículo dinâmico de Hover (acompanha o mouse sobre os pontos)
    const hGroup = new THREE.Group();
    const hRingGeo = new THREE.RingGeometry(0.8, 1.3, 24);
    const hRingMat = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.9,
      depthTest: false
    });
    hGroup.add(new THREE.Mesh(hRingGeo, hRingMat));
    hGroup.renderOrder = 998;
    hGroup.visible = false;
    scene.add(hGroup);
    hoverReticle = hGroup;

    // 2. Retículo fixado (ao clicar no ponto)
    const pGroup = new THREE.Group();
    const pRingGeo = new THREE.RingGeometry(1.2, 1.8, 32);
    const pRingMat = new THREE.MeshBasicMaterial({
      color: 0x38bdf8,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.9,
      depthTest: false
    });
    pGroup.add(new THREE.Mesh(pRingGeo, pRingMat));

    const pDotGeo = new THREE.CircleGeometry(0.35, 16);
    const pDotMat = new THREE.MeshBasicMaterial({
      color: 0x34d399,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.95,
      depthTest: false
    });
    pGroup.add(new THREE.Mesh(pDotGeo, pDotMat));

    pGroup.renderOrder = 999;
    pGroup.visible = false;
    scene.add(pGroup);
    pinnedReticle = pGroup;
  }

  function toggleHoverMode(enable) {
    isHoverEnabled = (enable !== undefined) ? enable : !isHoverEnabled;

    if (btnToggleHover) {
      btnToggleHover.classList.toggle('active', isHoverEnabled);
    }
    if (hudPlotlyBadge) {
      hudPlotlyBadge.style.opacity = isHoverEnabled ? '1' : '0.4';
      hudPlotlyBadge.querySelector('span:last-child').textContent = isHoverEnabled ? 'Hover Plotly Ativo' : 'Hover Pausado';
    }

    if (!isHoverEnabled) {
      hidePlotlyHover();
    }
  }

  function openDocInStudio(docItem) {
    if (!docItem) return;

    let existing = labelingDataset.find(d => String(d.id) === String(docItem.id));
    if (!existing) {
      labelingDataset.unshift(docItem);
      coordsIdxToDoc.set(docItem.coords_idx, docItem);
      existing = docItem;
    }

    let idx = studioQueue.findIndex(d => String(d.id) === String(docItem.id));
    if (idx === -1) {
      studioQueue.unshift(existing);
      studioIndex = 0;
    } else {
      studioIndex = idx;
    }

    switchView('studio');
    updateStudioView();
  }

  function resolveDocInfo(coordsIdx) {
    if (!pointCloud || coordsIdx === undefined || coordsIdx < 0 || coordsIdx >= nPoints) return null;

    const pos = pointsGeometry.attributes.position.array;
    const x = pos[coordsIdx * 3 + 0];
    const y = pos[coordsIdx * 3 + 1];
    const z = pos[coordsIdx * 3 + 2];

    let docId = docIds ? String(docIds[coordsIdx]) : String(coordsIdx);
    let className = 'Não Mapeado';
    let classHex = '#52525b';

    const cid = classIds ? classIds[coordsIdx] : 0;
    if (metadata.palette[cid]) {
      className = metadata.palette[cid].name;
      classHex = metadata.palette[cid].hex;
    }

    if (userAnnotations[docId]) {
      className = userAnnotations[docId];
      const pal = getClassPaletteByName(className);
      classHex = pal.hex;
    }

    const datasetDoc = coordsIdxToDoc.get(coordsIdx);

    let supplier = 'Não informado no lote reduzido';
    let subcota = 'Não informada no lote reduzido';
    let ano = '-';
    let proximity = '-';
    let url = null;

    if (datasetDoc) {
      docId = datasetDoc.id;
      className = datasetDoc.class;
      const pal = getClassPaletteByName(className);
      classHex = pal.hex;
      supplier = datasetDoc.fornecedor || 'N/A';
      subcota = datasetDoc.subcota || 'N/A';
      ano = datasetDoc.ano || '-';
      const simPct = (datasetDoc.similarity * 100).toFixed(1);
      proximity = `${datasetDoc.closest_cluster} (${simPct}%)`;
      url = datasetDoc.url;
    } else {
      if (urlPrefixes && urlIndices && docId) {
        const pref = urlPrefixes[urlIndices[coordsIdx]];
        if (pref) {
          url = `https://www.camara.leg.br/cota-parlamentar/documentos/publ/${pref}/${docId}.pdf`;
          const parts = pref.split('/');
          if (parts.length > 1) ano = parts[1];
        }
      }
    }

    if (!url && docId) {
      url = `https://www.camara.leg.br/cota-parlamentar/documentos/publ/${docId}.pdf`;
    }

    return {
      coordsIdx,
      docId,
      className,
      classHex,
      supplier,
      subcota,
      ano,
      proximity,
      url,
      x, y, z
    };
  }

  function showPlotlyHover(coordsIdx, clientX, clientY) {
    if (!isHoverEnabled) return;
    hoveredCoordsIdx = coordsIdx;
    const info = resolveDocInfo(coordsIdx);
    if (!info) return;

    if (hoverReticle) {
      hoverReticle.position.set(info.x, info.y, info.z);
      hoverReticle.children[0].material.color.set(info.classHex);
      hoverReticle.visible = true;
    }

    plotlyHoverTooltip.style.borderLeftColor = info.classHex;
    hoverClassDot.style.background = info.classHex;
    hoverClassDot.style.boxShadow = `0 0 8px ${info.classHex}`;
    hoverClassName.textContent = info.className;
    hoverDocId.textContent = `#${info.docId}`;

    hoverSupplier.textContent = info.supplier;
    hoverSubcota.textContent = info.subcota;
    hoverProximity.textContent = info.proximity !== '-' ? info.proximity : `Ano: ${info.ano}`;
    hoverCoords.textContent = `x: ${info.x.toFixed(1)}, y: ${info.y.toFixed(1)}, z: ${info.z.toFixed(1)}`;

    let left = clientX + 16;
    let top = clientY;

    if (left + 320 > window.innerWidth) {
      left = clientX - 330;
    }
    if (top + 90 > window.innerHeight) {
      top = window.innerHeight - 95;
    } else if (top - 80 < 60) {
      top = 70;
    }

    plotlyHoverTooltip.style.left = `${left}px`;
    plotlyHoverTooltip.style.top = `${top}px`;
    plotlyHoverTooltip.classList.remove('hidden');
  }

  function hidePlotlyHover() {
    hoveredCoordsIdx = null;
    if (plotlyHoverTooltip) plotlyHoverTooltip.classList.add('hidden');
    if (hoverReticle) hoverReticle.visible = false;
  }

  function pinPlotlyDoc(coordsIdx) {
    pinnedCoordsIdx = coordsIdx;
    const info = resolveDocInfo(coordsIdx);
    if (!info) return;

    if (pinnedReticle) {
      pinnedReticle.position.set(info.x, info.y, info.z);
      pinnedReticle.children[0].material.color.set(info.classHex);
      pinnedReticle.visible = true;
    }

    plotlyPinnedCard.style.borderLeftColor = info.classHex;
    pinnedClassDot.style.background = info.classHex;
    pinnedClassDot.style.boxShadow = `0 0 8px ${info.classHex}`;
    pinnedDocTitle.textContent = `Doc #${info.docId}`;
    pinnedClassBadge.textContent = info.className;
    pinnedClassBadge.style.color = info.classHex;
    pinnedClassBadge.style.borderColor = info.classHex;
    pinnedClassBadge.style.backgroundColor = `${info.classHex}22`;

    pinnedSupplier.textContent = info.supplier;
    pinnedSupplier.title = info.supplier;
    pinnedSubcota.textContent = info.subcota;
    pinnedSubcota.title = info.subcota;
    pinnedYear.textContent = info.ano;
    pinnedCluster.textContent = info.proximity !== '-' ? info.proximity : 'Não Mapeado';
    pinnedCoords.textContent = `X: ${info.x.toFixed(2)} | Y: ${info.y.toFixed(2)} | Z: ${info.z.toFixed(2)}`;

    if (info.url) {
      pinnedCamaraLink.href = info.url;
      pinnedCamaraLink.style.display = 'inline-flex';
      pinnedGoogleLink.href = `https://docs.google.com/viewer?url=${encodeURIComponent(info.url)}&embedded=true`;
      pinnedGoogleLink.style.display = 'inline-flex';
    } else {
      pinnedCamaraLink.style.display = 'none';
      pinnedGoogleLink.style.display = 'none';
    }

    btnPinnedOpenStudio.onclick = () => {
      openDocInStudio({
        id: String(info.docId),
        name: `${info.docId}.pdf`,
        class: info.className,
        fornecedor: info.supplier,
        subcota: info.subcota,
        ano: info.ano,
        url: info.url,
        closest_cluster: info.proximity !== '-' ? info.proximity.split(' ')[0] : info.className,
        similarity: 1.0,
        coords_idx: coordsIdx,
        status: 'rotulado'
      });
    };

    plotlyPinnedCard.classList.remove('hidden');
    hidePlotlyHover();
  }

  function unpinPlotlyDoc() {
    pinnedCoordsIdx = null;
    if (plotlyPinnedCard) plotlyPinnedCard.classList.add('hidden');
    if (pinnedReticle) pinnedReticle.visible = false;
  }

  // =========================================================================
  // MOTOR DA TABELA DE DADOS & EXPORTADOR CSV
  // =========================================================================
  function populateTableClassFilter() {
    tableFilterClass.innerHTML = `
      <option value="all">Todas as Classes</option>
      <option value="unmapped">Somente Não Mapeados</option>
    `;

    Object.values(metadata.palette)
      .filter(p => p.name !== 'Não Mapeado')
      .sort((a, b) => a.name.localeCompare(b.name))
      .forEach(p => {
        const opt = document.createElement('option');
        opt.value = p.name;
        opt.textContent = `${p.name} (${formatNumber(p.count)})`;
        tableFilterClass.appendChild(opt);
      });
  }

  function applyTableFilters() {
    const q = tableSearchInput.value.trim().toLowerCase();
    const classFilter = tableFilterClass.value;
    const sortVal = tableSortSelect.value;

    tableFiltered = labelingDataset.filter(d => {
      // Filtro de texto
      if (q) {
        const matchId = d.id.toLowerCase().includes(q);
        const matchForn = (d.fornecedor || '').toLowerCase().includes(q);
        const matchSub = (d.subcota || '').toLowerCase().includes(q);
        if (!matchId && !matchForn && !matchSub) return false;
      }

      // Filtro de classe
      if (classFilter === 'unmapped') {
        if (d.class !== 'Não Mapeado') return false;
      } else if (classFilter !== 'all') {
        if (d.class !== classFilter) return false;
      }

      return true;
    });

    // Ordenação
    tableFiltered.sort((a, b) => {
      if (sortVal === 'sim_desc') return b.similarity - a.similarity;
      if (sortVal === 'sim_asc') return a.similarity - b.similarity;
      if (sortVal === 'id_asc') return a.id.localeCompare(b.id);
      if (sortVal === 'forn_asc') return (a.fornecedor || '').localeCompare(b.fornecedor || '');
      return 0;
    });

    tableTotalCount.textContent = `${formatNumber(tableFiltered.length)} registros`;
    tablePage = 1;
    renderTablePage();
  }

  function renderTablePage() {
    tableBody.innerHTML = '';
    const totalPages = Math.max(1, Math.ceil(tableFiltered.length / tablePageSize));
    if (tablePage > totalPages) tablePage = totalPages;

    pageInfoText.textContent = `Página ${tablePage} de ${totalPages}`;
    pageCurrBtn.textContent = tablePage;

    const start = (tablePage - 1) * tablePageSize;
    const end = Math.min(start + tablePageSize, tableFiltered.length);
    const pageItems = tableFiltered.slice(start, end);

    pageItems.forEach(doc => {
      const tr = document.createElement('tr');
      const pal = getClassPaletteByName(doc.class);
      const proxSim = (doc.similarity * 100).toFixed(1);

      tr.innerHTML = `
        <td style="font-family: monospace; font-weight: 700; color: #fff;">#${doc.id}</td>
        <td>
          <span class="table-badge" style="color: ${pal.hex}; border: 1px solid ${pal.hex}; background: ${pal.hex}18;">
            ${doc.class}
          </span>
        </td>
        <td style="max-width: 260px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${doc.fornecedor}">
          ${doc.fornecedor || 'N/A'}
        </td>
        <td style="max-width: 240px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text-dim);" title="${doc.subcota}">
          ${doc.subcota || 'N/A'}
        </td>
        <td>
          <span style="color: #94a3b8;">${doc.closest_cluster}</span>
        </td>
        <td style="font-family: monospace; color: ${doc.similarity >= 0.95 ? 'var(--accent-green)' : '#94a3b8'};">
          ${proxSim}%
        </td>
        <td>
          <div class="table-actions">
            <button class="table-action-btn btn-label-here" title="Abrir no Estúdio de Rotulação">🏷️ Rotular</button>
            <a href="${doc.url}" target="_blank" rel="noopener noreferrer" class="table-action-btn" title="Abrir Documento Oficial">↗ PDF</a>
            <button class="table-action-btn btn-view-3d-here" title="Ver Ponto no Espaço 3D">🎯 3D</button>
          </div>
        </td>
      `;

      tr.querySelector('.btn-label-here').addEventListener('click', () => {
        studioQueue = [doc];
        studioIndex = 0;
        switchView('studio');
      });

      tr.querySelector('.btn-view-3d-here').addEventListener('click', () => {
        focusDocIn3D(doc.coords_idx);
      });

      tableBody.appendChild(tr);
    });
  }

  // --- Exportação Nativa CSV & JSON ---
  function exportCSV() {
    if (!labelingDataset || labelingDataset.length === 0) {
      alert('Nenhum dado disponível para exportação.');
      return;
    }

    showLoading('Gerando arquivo CSV completo...');
    setTimeout(() => {
      const headers = [
        'doc_id',
        'doc_name',
        'classe',
        'fornecedor',
        'subcota',
        'ano',
        'url_documento',
        'url_google_viewer',
        'cluster_mais_proximo',
        'similaridade',
        'status_rotulacao'
      ];

      const rows = [headers.join(',')];

      labelingDataset.forEach(d => {
        const gViewUrl = `https://docs.google.com/viewer?url=${encodeURIComponent(d.url)}`;
        const cleanForn = `"${(d.fornecedor || '').replace(/"/g, '""')}"`;
        const cleanSub = `"${(d.subcota || '').replace(/"/g, '""')}"`;
        const row = [
          d.id,
          d.name,
          d.class,
          cleanForn,
          cleanSub,
          d.ano,
          `"${d.url}"`,
          `"${gViewUrl}"`,
          d.closest_cluster,
          d.similarity,
          d.status || 'rotulado'
        ];
        rows.push(row.join(','));
      });

      const csvContent = '\uFEFF' + rows.join('\r\n');
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `ceap_documentos_rotulados_${new Date().toISOString().slice(0, 10)}.csv`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      hideLoading();
    }, 50);
  }

  function exportJSON() {
    showLoading('Gerando JSON de anotações...');
    setTimeout(() => {
      const map = {};
      labelingDataset.forEach(d => {
        if (d.class && d.class !== 'Não Mapeado') {
          map[d.id] = d.class;
        }
      });

      const jsonStr = JSON.stringify(map, null, 2);
      const blob = new Blob([jsonStr], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `annotations_history_${new Date().toISOString().slice(0, 10)}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      hideLoading();
    }, 50);
  }

  // =========================================================================
  // SETUP DE EVENT LISTENERS
  // =========================================================================
  function setupEventListeners() {
    // 1. Navegação de Telas
    btnView3D.addEventListener('click', () => switchView('3d'));
    btnViewStudio.addEventListener('click', () => switchView('studio'));
    btnViewTable.addEventListener('click', () => switchView('table'));

    // 2. Exportação Global
    btnExportCSV.addEventListener('click', exportCSV);
    btnStudioExportCSV.addEventListener('click', exportCSV);
    btnTableDownloadCSV.addEventListener('click', exportCSV);
    btnTableDownloadJSON.addEventListener('click', exportJSON);

    // 3. Estúdio de Rotulação
    studioQueueSelect.addEventListener('change', filterStudioQueue);
    studioPrevBtn.addEventListener('click', prevDoc);
    studioNextBtn.addEventListener('click', nextDoc);
    studioSkipBtn.addEventListener('click', nextDoc);

    docReloadViewer.addEventListener('click', () => {
      if (currentStudioDoc) {
        iframeLoader.style.display = 'flex';
        docIframe.src = `https://docs.google.com/viewer?url=${encodeURIComponent(currentStudioDoc.url)}&embedded=true`;
      }
    });

    docFocus3D.addEventListener('click', () => {
      if (currentStudioDoc) {
        focusDocIn3D(currentStudioDoc.coords_idx);
      }
    });

    // 4. Criação Dinâmica de Classes
    btnToggleAddClass.addEventListener('click', () => {
      newClassForm.classList.toggle('hidden');
      if (!newClassForm.classList.contains('hidden')) {
        newClassName.focus();
      }
    });

    btnCancelAddClass.addEventListener('click', () => {
      newClassForm.classList.add('hidden');
      newClassName.value = '';
    });

    btnConfirmAddClass.addEventListener('click', () => {
      addNewCustomClass(newClassName.value, newClassColor.value);
    });

    newClassName.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        addNewCustomClass(newClassName.value, newClassColor.value);
      }
    });

    // 5. Atalhos de Teclado no Estúdio de Rotulação
    window.addEventListener('keydown', (e) => {
      // Ignorar atalhos se estiver digitando em inputs
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

      if (!viewStudio.classList.contains('active')) return;

      if (e.key === 'ArrowRight') {
        e.preventDefault();
        nextDoc();
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        prevDoc();
      } else if (e.code === 'Space') {
        e.preventDefault();
        nextDoc();
      } else {
        const k = e.key.toUpperCase();
        if (shortcutToClass[k]) {
          e.preventDefault();
          assignLabel(shortcutToClass[k]);
        }
      }
    });

    // 6. Controles da Tabela
    tableSearchInput.addEventListener('input', applyTableFilters);
    tableFilterClass.addEventListener('change', applyTableFilters);
    tableSortSelect.addEventListener('change', applyTableFilters);

    pageFirstBtn.addEventListener('click', () => { tablePage = 1; renderTablePage(); });
    pagePrevBtn.addEventListener('click', () => { if (tablePage > 1) { tablePage--; renderTablePage(); } });
    pageNextBtn.addEventListener('click', () => {
      const maxPages = Math.ceil(tableFiltered.length / tablePageSize);
      if (tablePage < maxPages) { tablePage++; renderTablePage(); }
    });
    pageLastBtn.addEventListener('click', () => {
      tablePage = Math.ceil(tableFiltered.length / tablePageSize);
      renderTablePage();
    });

    // 7. Controles 3D e Three.js
    modelTabBtns.forEach((btn) => {
      btn.addEventListener('click', () => {
        const model = btn.getAttribute('data-model');
        switchModel(model);
      });
    });

    sliderSize.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      sizeVal.textContent = `${val.toFixed(1)}px`;
      if (pointsMaterial) pointsMaterial.uniforms.uGlobalSize.value = val;
    });

    sliderUnmapped.addEventListener('input', (e) => {
      const val = parseInt(e.target.value, 10);
      unmappedVal.textContent = `${val}%`;
      if (pointsMaterial) pointsMaterial.uniforms.uUnmappedAlpha.value = val / 100.0;
    });

    btnShowAll.addEventListener('click', () => {
      isHideUnmapped = false;
      btnHideUnmapped.textContent = 'Ocultar Não Mapeados';
      showAllClasses();
    });

    btnHideUnmapped.addEventListener('click', () => {
      isHideUnmapped = !isHideUnmapped;
      classVisibility[0] = isHideUnmapped ? 0.0 : 1.0;
      btnHideUnmapped.textContent = isHideUnmapped ? 'Mostrar Não Mapeados' : 'Ocultar Não Mapeados';
      updateLegendVisuals();
    });

    classSearch.addEventListener('input', (e) => {
      const q = e.target.value.trim().toLowerCase();
      const items = legendList.querySelectorAll('.legend-item');
      items.forEach((item) => {
        const name = item.querySelector('.class-name').textContent.toLowerCase();
        item.style.display = name.includes(q) ? 'flex' : 'none';
      });
    });

    btnResetCam.addEventListener('click', () => {
      camera.position.set(0, 30, 95);
      controls.target.set(0, 0, 0);
    });

    btnToggleAutorotate.addEventListener('click', () => {
      controls.autoRotate = !controls.autoRotate;
      btnToggleAutorotate.classList.toggle('active', controls.autoRotate);
    });

    btnFullscreen.addEventListener('click', () => {
      if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().catch(() => {});
      } else {
        document.exitFullscreen().catch(() => {});
      }
    });

    btnTogglePanel.addEventListener('click', () => {
      sidePanel.classList.toggle('collapsed');
      btnTogglePanel.classList.toggle('active', !sidePanel.classList.contains('collapsed'));
    });

    // 8. Inspeção 3D Estilo Plotly (Hover em tempo real + Clique para fixar)
    if (btnToggleHover) {
      btnToggleHover.addEventListener('click', () => toggleHoverMode());
    }

    if (btnClosePinned) {
      btnClosePinned.addEventListener('click', unpinPlotlyDoc);
    }

    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    let pointerStartPos = { x: 0, y: 0, time: 0 };

    // Detectar início e fim de órbita para pausar hover durante rotação da câmera (zero lag!)
    controls.addEventListener('start', () => {
      isOrbitDragging = true;
      hidePlotlyHover();
    });

    controls.addEventListener('end', () => {
      isOrbitDragging = false;
    });

    // Hover estilo Plotly: segue o mouse e exibe o tooltip ao passar sobre qualquer ponto
    renderer.domElement.addEventListener('pointermove', (e) => {
      if (isOrbitDragging || !isHoverEnabled || !pointCloud) return;

      const now = performance.now();
      if (now - lastHoverTimestamp < 25) return; // Limita a ~40 FPS para máximo desempenho
      lastHoverTimestamp = now;

      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      const camDist = camera.position.distanceTo(controls.target);
      raycaster.params.Points.threshold = Math.max(1.2, Math.min(4.5, camDist * 0.022));
      raycaster.setFromCamera(mouse, camera);

      const intersects = raycaster.intersectObject(pointCloud);
      if (intersects && intersects.length > 0) {
        showPlotlyHover(intersects[0].index, e.clientX, e.clientY);
      } else {
        hidePlotlyHover();
      }
    });

    renderer.domElement.addEventListener('pointerleave', hidePlotlyHover);

    // Clique simples (sem arrasto) para fixar o ponto e abrir opções
    renderer.domElement.addEventListener('pointerdown', (e) => {
      pointerStartPos = { x: e.clientX, y: e.clientY, time: Date.now() };
    });

    renderer.domElement.addEventListener('pointerup', (e) => {
      const dx = Math.abs(e.clientX - pointerStartPos.x);
      const dy = Math.abs(e.clientY - pointerStartPos.y);
      const dt = Date.now() - pointerStartPos.time;

      // Se o usuário arrastou a câmera (órbita / pan), não tratar como clique
      if (dx > 5 || dy > 5 || dt > 1000) return;

      if (hoveredCoordsIdx !== null) {
        pinPlotlyDoc(hoveredCoordsIdx);
      } else if (pinnedCoordsIdx !== null) {
        unpinPlotlyDoc();
      }
    });

    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        unpinPlotlyDoc();
        hidePlotlyHover();
      }
    });

    window.addEventListener('resize', () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    });
  }

  // --- Loop de Animação 3D ---
  function animate() {
    requestAnimationFrame(animate);
    controls.update();

    if (hoverReticle && hoverReticle.visible) {
      hoverReticle.quaternion.copy(camera.quaternion);
      const camDist = camera.position.distanceTo(hoverReticle.position);
      const scale = Math.max(0.6, camDist * 0.016);
      hoverReticle.scale.set(scale, scale, scale);
    }

    if (pinnedReticle && pinnedReticle.visible) {
      pinnedReticle.quaternion.copy(camera.quaternion);
      const camDist = camera.position.distanceTo(pinnedReticle.position);
      const scale = Math.max(0.6, camDist * 0.016);
      const pulse = 1.0 + 0.12 * Math.sin(Date.now() * 0.007);
      pinnedReticle.scale.set(scale * pulse, scale * pulse, scale * pulse);
    }

    renderer.render(scene, camera);
  }

  // =========================================================================
  // INICIALIZAÇÃO GERAL
  // =========================================================================
  async function main() {
    showLoading('Carregando metadados do CEAP...');
    try {
      metadata = await loadMetadata();
      nPoints = metadata.n_points;

      // Carregar classes customizadas salvas pelo usuário
      Object.entries(customClasses).forEach(([name, palItem]) => {
        const cid = Object.keys(metadata.palette).length;
        metadata.palette[cid] = palItem;
      });

      pointCountBadge.textContent = '1.34M docs • 1.97M págs';
      totalAnnotatedCount.textContent = `${formatNumber(metadata.total_annotated || 22049)} docs`;

      buildLegend();
      setupEventListeners();

      showLoading('Carregando classes dos 1.34M documentos...');
      classIds = await loadClasses();

      // Aplicar anotações locais salvas pelo usuário se houver
      Object.entries(userAnnotations).forEach(([stem, newClass]) => {
        const newCid = getClassIdByName(newClass);
        // Atualizar se soubermos o índice
      });

      // Carregar modelo 3D padrão
      await switchModel(metadata.default_model || 'metric');

      // Carregar dataset de rotulação assincronamente em background
      showLoading('Carregando dataset do Estúdio de Rotulação...');
      labelingDataset = await loadLabelingDataset();

      // Aplicar anotações locais ao dataset
      labelingDataset.forEach(d => {
        if (userAnnotations[d.id]) {
          d.class = userAnnotations[d.id];
          d.status = 'rotulado';
        }
        if (d.coords_idx !== undefined) {
          coordsIdxToDoc.set(d.coords_idx, d);
        }
      });

      // Carregar índice geral de IDs e mapeamento de URLs em background
      loadDocIds().then(ids => {
        if (ids) {
          docIds = ids;
          console.log(`Índice de ${formatNumber(ids.length)} IDs de documentos carregado.`);
        }
      }).catch(err => console.warn('Erro ao carregar doc_ids:', err));

      loadUrlMapping().then(() => {
        console.log('Mapeamento universal de URLs da Câmara carregado.');
      }).catch(err => console.warn('Erro ao carregar url_mapping:', err));

      createPlotlyReticles();
      updatePendingBadge();
      populateTableClassFilter();
      filterStudioQueue();

      hideLoading();
      animate();
    } catch (err) {
      console.error('Falha na inicialização:', err);
      alert('Erro ao carregar dados da visualização: ' + err.message);
      hideLoading();
    }
  }

  main();
})();
