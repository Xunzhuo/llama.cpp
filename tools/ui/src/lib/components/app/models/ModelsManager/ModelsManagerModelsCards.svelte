<script lang="ts">
	import ModelDownloadProgressBar from '../ModelDownloadProgressBar.svelte';
	import ModelsManagerTableToolbar from './ModelsManagerTableToolbar.svelte';
	import { modelRowActions } from './row-actions';
	import {
		canLoadOption,
		downloadProgressFor,
		type ModelQuantGroup,
		type ModelsTableGroup
	} from './utils';
	import {
		CheckCircle2,
		ChevronDown,
		ChevronUp,
		Download,
		EyeOff,
		Heart,
		MoreHorizontal,
		Power
	} from '@lucide/svelte';
	import {
		CollapsibleRegion,
		DropdownMenuActions,
		GroupedList,
		type GroupedListGroup,
		GroupedListUnit,
		Logo,
		ModelAvatar,
		ModelId,
		ModelsSection
	} from '$lib/components/app';
	import { DialogConfirmDownload } from '$lib/components/app/dialogs';
	import { Badge } from '$lib/components/ui/badge';
	import * as Card from '$lib/components/ui/card';
	import {
		FAMILY_ROW_WINDOW,
		type ModalityKey,
		MODEL_ROW_WINDOW,
		SETTINGS_KEYS
	} from '$lib/constants';
	import {
		ModelCapability,
		ModelDownloadConfirmAction,
		ModelGroupKind,
		ModelRowDownloadState,
		ModelsTableGroupKind
	} from '$lib/enums';
	import { modelsStore, settingsStore } from '$lib/stores';
	import type { ModelOption } from '$lib/types/models';
	import { groupModelFamilies, type ModelFamilyGroup } from '$lib/utils/model-families';
	import type { Snippet } from 'svelte';
	import { SvelteSet } from 'svelte/reactivity';

	interface Props {
		/** Capabilities a model must have every one of. */
		capabilities?: ModelCapability[];
		/** Smallest context a model must support; 0 keeps every model. */
		contextLimit?: number;
		filter?: string;
		groups: ModelsTableGroup[];
		isFavorite: (option: ModelOption) => boolean;
		onSelect: (option: ModelOption) => void;
		/** Modalities a model must support at least one of. */
		modalities?: ModalityKey[];
		selectedId: string | null;
		/** Rendered at the toolbar's right end, past the filters. */
		toolbarEnd?: Snippet;
	}

	let {
		capabilities = $bindable<ModelCapability[]>([]),
		contextLimit = $bindable(0),
		filter = $bindable(''),
		groups,
		isFavorite,
		modalities = $bindable<ModalityKey[]>([]),
		onSelect,
		selectedId,
		toolbarEnd
	}: Props = $props();

	let isEmpty = $derived(groups.every((group) => group.items.length === 0));
	let hasFilters = $derived(contextLimit > 0 || modalities.length > 0 || capabilities.length > 0);

	/** Noun the show-more row counts in, per unit of the grouped list. */
	const SHOW_MORE_NOUNS: Record<GroupedListUnit, string> = {
		[GroupedListUnit.ENTRIES]: 'models',
		[GroupedListUnit.GROUPS]: 'families'
	};

	let pendingDelete = $state('');
	let deleteOpen = $state(false);
	/** Repos whose quants are folded away; the rest show them. */
	const collapsedQuants = new SvelteSet<string>();
	/** Sections that list their models straight, without folding them into families. */
	const FLAT_SECTIONS = new Set<ModelsTableGroup['kind']>([
		ModelsTableGroupKind.DOWNLOADING,
		ModelsTableGroupKind.FAVORITES,
		ModelsTableGroupKind.LOADED,
		ModelsTableGroupKind.SELECTED
	]);
	let sections = $derived(
		groups.map((group) => {
			const flat =
				FLAT_SECTIONS.has(group.kind) ||
				!settingsStore.config[SETTINGS_KEYS.GROUP_MODELS_BY_FAMILY];

			return {
				...group,
				families: flat ? [] : groupModelFamilies(group.items, (entry) => entry.base.model),
				flat
			};
		})
	);

	function familyGroups(
		group: (typeof sections)[number]
	): GroupedListGroup<ModelFamilyGroup<ModelQuantGroup>, ModelQuantGroup>[] {
		return group.families.map((family) => ({
			entries: family.entries,
			group: family,
			key: `${group.key}::${family.key}`
		}));
	}

	// cancel is confirmed once for the whole list, so one dialog serves every card
	function requestDelete(option: ModelOption): void {
		pendingDelete = option.model;
		deleteOpen = true;
	}

	function toggleQuants(key: string): void {
		if (collapsedQuants.has(key)) {
			collapsedQuants.delete(key);
		} else {
			collapsedQuants.add(key);
		}
	}
</script>

{#snippet familyCard({
	expanded,
	group: family,
	toggle
}: {
	expanded: boolean;
	group: ModelFamilyGroup<ModelQuantGroup>;
	toggle: () => void;
})}
	<button
		class="flex w-full cursor-pointer items-center gap-2.5 py-2.5 text-left transition hover:bg-muted/40"
		onclick={toggle}
		type="button"
	>
		<ModelAvatar
			option={family.entries[0].base}
			showBaseModelAvatar
			showQuantBadge={false}
			size="size-7"
		/>

		<span class="min-w-0 flex-1 truncate text-sm font-medium">{family.label}</span>

		<span class="text-xs text-muted-foreground">
			{family.entries.length} model{family.entries.length === 1 ? '' : 's'}
		</span>

		{#if expanded}
			<ChevronUp class="h-4 w-4 shrink-0 text-muted-foreground" />
		{:else}
			<ChevronDown class="h-4 w-4 shrink-0 text-muted-foreground" />
		{/if}
	</button>
{/snippet}

{#snippet quantCard(option: ModelOption)}
	{@const favorite = isFavorite(option)}
	{@const isHidden = modelsStore.isHidden(option.id)}
	{@const quant = option.parsedId?.quantization ?? option.model}

	<div
		class={[
			'relative rounded-lg transition hover:bg-muted/50',
			selectedId === option.id && 'bg-accent/50',
			isHidden && 'opacity-60'
		]}
	>
		<!-- the card holds an actions menu, so only the quant itself is the button -->
		<button
			class="flex w-full cursor-pointer items-center gap-2 py-2.5 pl-2.5 pr-1.5 text-left"
			onclick={() => onSelect(option)}
			type="button"
		>
			<Badge class="h-5 shrink-0 px-1.5 text-[10px]" variant="secondary">{quant}</Badge>

			<span class="min-w-0 flex-1 truncate text-sm text-muted-foreground">{option.model}</span>

			<DropdownMenuActions
				actions={modelRowActions(
					option,
					{ canLoad: canLoadOption(option), favorite, isHidden },
					{ onDelete: requestDelete }
				)}
				align="end"
				triggerIcon={MoreHorizontal}
				triggerTooltip="Model actions"
			/>
		</button>
	</div>
{/snippet}

{#snippet modelCard(entry: ModelQuantGroup)}
	{@const option = entry.base}
	{@const favorite = isFavorite(option)}
	{@const isHidden = modelsStore.isHidden(option.id)}
	{@const download = modelsStore.status.isDownloadPaused(option.model)
		? ModelRowDownloadState.PAUSED
		: modelsStore.status.isDownloadInProgress(option.model)
			? ModelRowDownloadState.DOWNLOADING
			: null}
	{@const downloadProgress = downloadProgressFor(option.model)}

	<Card.Root
		class="relative mb-2 !gap-3 bg-muted/30 p-4 {selectedId === option.id
			? 'border-accent! bg-accent/30'
			: ''} {isHidden ? 'opacity-60' : ''}"
	>
		<div class="flex items-center gap-3">
			<!-- the card holds an actions menu, so only the model itself is the button -->
			<button
				class="flex min-w-0 flex-1 cursor-pointer items-center gap-3 text-left"
				onclick={() => onSelect(option)}
				type="button"
			>
				<ModelAvatar {option} size="size-9" />

				<ModelId
					aliases={option.aliases}
					class="min-w-0 flex-1"
					draftSidecars={option.draftSidecars}
					modalities={option.modalities}
					modelId={option.model}
					stackId
					tags={option.tags}
					title={option.model}
				/>
			</button>

			<DropdownMenuActions
				actions={modelRowActions(
					option,
					{ canLoad: canLoadOption(option), download, favorite, isHidden },
					{ onDelete: requestDelete }
				)}
				align="end"
				triggerIcon={MoreHorizontal}
				triggerTooltip="Model actions"
			/>
		</div>

		{#if download}
			<ModelDownloadProgressBar
				downloadedBytes={downloadProgress?.downloadedBytes ?? 0}
				overlay
				totalBytes={downloadProgress?.totalBytes ?? 0}
			/>
		{/if}
	</Card.Root>
{/snippet}

{#snippet repoCard(entry: ModelQuantGroup)}
	{@const expanded = !collapsedQuants.has(entry.key)}
	{@const groupLabel =
		entry.kind === ModelGroupKind.VARIANTS
			? `${entry.quants.length} variants`
			: `${entry.quants.length} quants available`}

	<Card.Root class="mb-2 overflow-hidden !gap-0 bg-muted/30 p-0!">
		<button
			class="flex w-full cursor-pointer items-center gap-3 px-4 py-3 text-left transition hover:bg-muted/40"
			onclick={() => toggleQuants(entry.key)}
			type="button"
		>
			<ModelAvatar option={entry.base} size="size-9" />

			<span class="min-w-0 flex-1">
				<ModelId
					aliases={entry.base.aliases}
					class="min-w-0"
					hideCapabilities
					hideModalities
					hideQuantization
					modalities={entry.base.modalities}
					modelId={entry.base.model}
					stackId
					tags={entry.base.tags}
					title={entry.base.model}
				/>

				<span class="block text-xs text-muted-foreground">{groupLabel}</span>
			</span>

			{#if expanded}
				<ChevronUp class="h-4 w-4 shrink-0 text-muted-foreground" />
			{:else}
				<ChevronDown class="h-4 w-4 shrink-0 text-muted-foreground" />
			{/if}
		</button>

		<CollapsibleRegion open={expanded}>
			<div class="flex flex-col gap-1 border-t border-border/30 p-2 dark:border-border/20">
				{#each entry.quants as quant (quant.id)}
					{@render quantCard(quant)}
				{/each}
			</div>
		</CollapsibleRegion>
	</Card.Root>
{/snippet}

{#snippet cardTree(entry: ModelQuantGroup)}
	{#if entry.quants.length > 1}
		{@render repoCard(entry)}
	{:else}
		{@render modelCard(entry)}
	{/if}
{/snippet}

{#snippet cardItem({ entry }: { depth: number; entry: ModelQuantGroup })}
	{@render cardTree(entry)}
{/snippet}

{#snippet showMore({
	count,
	onMore,
	unit
}: {
	count: number;
	onMore: () => void;
	unit: GroupedListUnit;
})}
	<div class="px-3 pb-1">
		<button
			class="w-full cursor-pointer rounded-md px-3 py-2.5 text-left text-sm text-muted-foreground transition hover:bg-muted/40"
			onclick={onMore}
			type="button"
		>
			Show {count} more {SHOW_MORE_NOUNS[unit]}
		</button>
	</div>
{/snippet}

<DialogConfirmDownload
	action={ModelDownloadConfirmAction.DELETE}
	onClose={() => (deleteOpen = false)}
	open={deleteOpen}
	repoWithTag={pendingDelete}
/>

<div class="flex h-full min-h-0 flex-col">
	<ModelsManagerTableToolbar
		bind:capabilities
		bind:contextLimit
		bind:filter
		bind:modalities
		{toolbarEnd}
	/>

	<div class="min-h-0 flex-1 overflow-y-auto">
		<div class="flex flex-col gap-2">
			{#each sections as group (group.key)}
				{#if group.items.length > 0}
					{#snippet groupIcon()}
						{#if group.kind === ModelsTableGroupKind.DOWNLOADING}
							<Download class="h-3.5 w-3.5 shrink-0" />
						{:else if group.kind === ModelsTableGroupKind.FAVORITES}
							<Heart class="h-3.5 w-3.5 shrink-0" />
						{:else if group.kind === ModelsTableGroupKind.LOADED}
							<Power class="h-3.5 w-3.5 shrink-0" />
						{:else if group.kind === ModelsTableGroupKind.SELECTED}
							<CheckCircle2 class="h-3.5 w-3.5 shrink-0" />
						{:else if group.kind === ModelsTableGroupKind.HIDDEN}
							<EyeOff class="h-3.5 w-3.5 shrink-0" />
						{:else if group.kind === ModelsTableGroupKind.LOCAL}
							<Logo class="shrink-0" style="--size: 0.875rem" />
						{/if}
					{/snippet}

					<ModelsSection
						count={group.items.length}
						defaultOpen={group.kind !== ModelsTableGroupKind.HIDDEN}
						icon={groupIcon}
						label={group.label}
						persistKey={group.key}
						sectionHeaderClass="m-0 px-1 py-2 text-[13px] font-semibold text-muted-foreground select-none"
						sticky
						stickyClass="sticky z-10 bg-background/90 backdrop-blur-lg"
					>
						<GroupedList
							group={familyCard}
							groupWindow={FAMILY_ROW_WINDOW}
							groups={group.flat ? null : familyGroups(group)}
							initialCollapsed={modelsStore.collapsedGroupsUnder(group.key)}
							item={cardItem}
							items={group.flat ? group.items : []}
							keyOf={(entry) => entry.key}
							more={showMore}
							onCollapsedChange={(key, collapsed) =>
								modelsStore.setGroupCollapsed(group.key, key, collapsed)}
							sectionWindow={MODEL_ROW_WINDOW}
							stickyClass="bg-background/90 backdrop-blur-lg"
							stickyStyle="top: calc(2.25rem - 1px)"
							weightOf={(entry) => entry.quants.length}
						/>
					</ModelsSection>
				{/if}
			{/each}
		</div>

		{#if isEmpty}
			<p class="px-4 py-10 text-center text-sm text-muted-foreground">
				{hasFilters ? 'No models match these filters.' : 'No models found.'}
			</p>
		{/if}
	</div>
</div>
