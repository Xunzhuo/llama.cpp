<script lang="ts">
	import ModelsSelectorDropdown from '$lib/components/app/models/ModelsSelector/ModelsSelectorDropdown.svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';

	let { currentModel = null }: { currentModel?: string | null } = $props();
</script>

<!-- The picker rows mount tooltips, which need the provider +layout.svelte supplies. -->
<Tooltip.Provider>
	<ModelsSelectorDropdown {currentModel} />
</Tooltip.Provider>
